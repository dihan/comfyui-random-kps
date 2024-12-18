"""InstantID Face implementation with face selection and fixed keypoint extraction"""
import torch
import numpy as np
import comfy.model_management
import torchvision.transforms as T
import sys
import os

# Add InstantID path to system path
INSTANTID_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ComfyUI_InstantID'))
if os.path.exists(INSTANTID_PATH):
    if INSTANTID_PATH not in sys.path:
        sys.path.append(INSTANTID_PATH)
        print(f"Added InstantID path: {INSTANTID_PATH}")
else:
    print(f"Warning: InstantID path not found at {INSTANTID_PATH}")

try:
    from custom_nodes.ComfyUI_InstantID.InstantID import (
        ApplyInstantIDAdvanced,
        draw_kps,
        tensor_to_image
    )
    print("Successfully imported InstantID components")
except ImportError as e:
    print(f"Error importing InstantID components: {str(e)}")
    raise

def extractFeatures(insightface, image, extract_kps=False, face_selection="largest"):
    """Enhanced extractFeatures with face selection support and debug logging"""
    face_img = tensor_to_image(image)
    out = []

    insightface.det_model.input_size = (640,640)  # reset detection size
    print(f"Face selection mode: {face_selection}")

    for i in range(face_img.shape[0]):
        batch_out = None
        for size in [(size, size) for size in range(640, 128, -64)]:
            insightface.det_model.input_size = size
            faces = insightface.get(face_img[i])
            
            if faces:
                # Print initial face detection info
                print(f"\nDetected {len(faces)} faces")
                
                # Sort faces by size (area of bounding box)
                faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))
                
                # Print face sizes for debugging
                for idx, face in enumerate(faces):
                    bbox = face['bbox']
                    area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
                    print(f"Face {idx}: Area = {area:.2f}, Bbox = {bbox}")
                
                # Select face based on selection parameter
                if face_selection == "smallest":
                    selected_face = faces[0]
                    print("Selected smallest face")
                elif face_selection == "medium" and len(faces) >= 3:
                    selected_face = faces[len(faces)//2]
                    print("Selected medium face")
                else:  # "largest" or default case
                    selected_face = faces[-1]
                    print(f"Selected largest face (default or {face_selection} mode)")
                
                # Print selected face info
                bbox = selected_face['bbox']
                area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
                print(f"Selected face bbox: {bbox}, area: {area:.2f}")
                
                if extract_kps:
                    # Generate keypoint visualization
                    kps_img = draw_kps(face_img[i], selected_face['kps'])
                    batch_out = T.ToTensor()(kps_img).permute(1, 2, 0)
                    print("Generated keypoint visualization")
                else:
                    # Extract face embedding
                    batch_out = torch.from_numpy(selected_face['embedding']).unsqueeze(0)
                    print("Extracted face embedding")

                if 640 not in size:
                    print(f"\033[33mINFO: InsightFace detection resolution lowered to {size}.\033[0m")
                break
        
        if batch_out is not None:
            out.append(batch_out)

    if out:
        if extract_kps:
            return torch.stack(out, dim=0).permute([0,2,3,1])
        else:
            return torch.stack(out, dim=0)
    
    print("No faces detected")
    return None

class InstantIDFace(ApplyInstantIDAdvanced):
    @classmethod
    def INPUT_TYPES(s):
        original_types = super().INPUT_TYPES()
        original_types["required"]["face_selection"] = (["largest", "smallest", "medium"], {"default": "largest"})
        return original_types

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("MODEL", "positive", "negative",)
    FUNCTION = "apply_instantid"
    CATEGORY = "InstantID"

    def apply_instantid(self, instantid, insightface, control_net, image, model, positive, negative, 
                       ip_weight, cn_strength, start_at, end_at, noise, face_selection,
                       image_kps=None, mask=None, combine_embeds='average'):
        
        dtype = comfy.model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32
        
        self.dtype = dtype
        self.device = comfy.model_management.get_torch_device()

        # Use enhanced extractFeatures with face selection
        face_embed = extractFeatures(insightface, image, face_selection=face_selection)
        if face_embed is None:
            raise Exception('Reference Image: No face detected.')

        face_kps = extractFeatures(insightface, 
                                 image_kps if image_kps is not None else image[0].unsqueeze(0), 
                                 extract_kps=True, 
                                 face_selection=face_selection)

        if face_kps is None:
            face_kps = torch.zeros_like(image) if image_kps is None else image_kps
            print(f"\033[33mWARNING: No face detected in the keypoints image!\033[0m")

        clip_embed = face_embed
        if clip_embed.shape[0] > 1:
            if combine_embeds == 'average':
                clip_embed = torch.mean(clip_embed, dim=0).unsqueeze(0)
            elif combine_embeds == 'norm average':
                clip_embed = torch.mean(clip_embed / torch.norm(clip_embed, dim=0, keepdim=True), dim=0).unsqueeze(0)

        if noise > 0:
            seed = int(torch.sum(clip_embed).item()) % 1000000007
            torch.manual_seed(seed)
            clip_embed_zeroed = noise * torch.rand_like(clip_embed)
        else:
            clip_embed_zeroed = torch.zeros_like(clip_embed)

        self.instantid = instantid
        self.instantid.to(self.device, dtype=self.dtype)

        image_prompt_embeds, uncond_image_prompt_embeds = self.instantid.get_image_embeds(
            clip_embed.to(self.device, dtype=self.dtype), 
            clip_embed_zeroed.to(self.device, dtype=self.dtype)
        )

        return super().apply_instantid(
            instantid=instantid,
            insightface=insightface,
            control_net=control_net,
            image=image,
            model=model,
            positive=positive,
            negative=negative,
            ip_weight=ip_weight,
            cn_strength=cn_strength,
            start_at=start_at,
            end_at=end_at,
            noise=noise,
            image_kps=image_kps,
            mask=mask,
            combine_embeds=combine_embeds
        )

# Node mappings
NODE_CLASS_MAPPINGS = {
    "InstantIDFace": InstantIDFace,
}

# Display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "InstantIDFace": "InstantID Face",
}