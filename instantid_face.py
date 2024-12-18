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
    """Enhanced extractFeatures with detailed face detection and logging"""
    face_img = tensor_to_image(image)
    out = []

    print(f"\nFace selection mode: {face_selection}")

    for i in range(face_img.shape[0]):
        batch_out = None
        all_faces = []
        have_valid_detection = False
        
        # Try multiple detection sizes to find all faces
        for size in [(640, 640), (512, 512), (384, 384)]:
            faces = insightface.get(face_img[i])
            if faces and len(faces) > 0:
                print(f"\nFound {len(faces)} faces at size {size}")
                for face in faces:
                    bbox = face['bbox']
                    area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
                    width = bbox[2]-bbox[0]
                    height = bbox[3]-bbox[1]
                    
                    # Basic validation
                    if width > 20 and height > 20:
                        print(f"Valid face found - Area: {area:.2f}, Bbox: {bbox}")
                        all_faces.append((area, face))
                        have_valid_detection = True
                    else:
                        print(f"Skipping small face - Area: {area:.2f}, Bbox: {bbox}")
        
        if have_valid_detection and all_faces:
            # Sort all detected faces by area (largest to smallest)
            all_faces.sort(key=lambda x: x[0], reverse=True)
            print(f"\nFound {len(all_faces)} valid faces:")
            for idx, (area, face) in enumerate(all_faces):
                print(f"Face {idx} - Area: {area:.2f}, Bbox: {face['bbox']}")
            
            # Select face based on mode
            if face_selection == "smallest" and len(all_faces) > 1:
                # Select the second largest face if multiple faces exist
                selected_area, selected_face = all_faces[1]
                print(f"Selected second largest face (area: {selected_area:.2f})")
            else:
                # Default to largest face
                selected_area, selected_face = all_faces[0]
                print(f"Selected largest face (area: {selected_area:.2f})")
            
            bbox = selected_face['bbox']
            print(f"Selected face bbox: {bbox}")
            
            if extract_kps:
                kps_img = draw_kps(face_img[i], selected_face['kps'])
                batch_out = T.ToTensor()(kps_img).permute(1, 2, 0)
                print("Generated keypoint visualization")
            else:
                batch_out = torch.from_numpy(selected_face['embedding']).unsqueeze(0)
                print("Extracted face embedding")
            
            out.append(batch_out)
        else:
            print("No valid faces detected in image")
            if extract_kps:
                return torch.zeros_like(image)
            continue

    if len(out) > 0:
        if extract_kps:
            return torch.stack(out, dim=0)
        else:
            return torch.stack(out, dim=0)
    else:
        if extract_kps:
            return torch.zeros_like(image)
        return None

class InstantIDFace(ApplyInstantIDAdvanced):
    @classmethod
    def INPUT_TYPES(s):
        original_types = super().INPUT_TYPES()
        original_types["required"]["face_selection"] = (["largest", "smallest"], {"default": "largest"})
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

        # Extract face features with consistent face selection
        face_embed = extractFeatures(insightface, image, face_selection=face_selection)
        if face_embed is None:
            print("\033[33mWARNING: No faces detected in reference image. Processing will be limited.\033[0m")
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

        # Extract keypoints using same face selection mode
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