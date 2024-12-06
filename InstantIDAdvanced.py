from .InstantID import (
    NODE_CLASS_MAPPINGS, 
    NODE_DISPLAY_NAME_MAPPINGS, 
    ApplyInstantIDAdvanced,
    draw_kps
)
from .utils import tensor_to_image
import torch
import torch.nn.functional as F
import numpy as np
import PIL.Image
import comfy.model_management


def extractFeatures(insightface, image, extract_kps=False, face_selection="largest"):
    face_img = tensor_to_image(image)
    out = []

    insightface.det_model.input_size = (640,640) # reset the detection size

    for i in range(face_img.shape[0]):
        for size in [(size, size) for size in range(640, 128, -64)]:
            insightface.det_model.input_size = size
            faces = insightface.get(face_img[i])
            if faces:
                # Sort faces by size
                faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))
                
                # Select face based on selection parameter
                if face_selection == "smallest":
                    face = faces[0]
                elif face_selection == "medium" and len(faces) > 2:
                    face = faces[len(faces)//2]
                else:  # "largest" or any other case
                    face = faces[-1]

                if extract_kps:
                    kps_img = draw_kps(face_img[i], face['kps'])
                    # Convert PIL Image to numpy array first
                    if isinstance(kps_img, PIL.Image.Image):
                        kps_img = np.array(kps_img)
                    # Convert numpy array to tensor
                    kps_tensor = torch.from_numpy(kps_img).float().permute(2, 0, 1) / 255.0
                    out.append(kps_tensor)
                else:
                    out.append(torch.from_numpy(face['embedding']).unsqueeze(0))

                if 640 not in size:
                    print(f"\033[33mINFO: InsightFace detection resolution lowered to {size}.\033[0m")
                break

    if out:
        if extract_kps:
            # Stack the already-converted tensors
            out = torch.stack(out, dim=0)
            # Convert to [batch, height, width, channels] format
            out = out.permute(0, 2, 3, 1)
        else:
            out = torch.stack(out, dim=0)
    else:
        if extract_kps:
            # Create blank tensor if no faces detected
            h, w, _ = face_img[0].shape
            out = torch.zeros((1, h, w, 3), dtype=torch.float32)
        else:
            out = None

    return out

class ApplyInstantIDAdvancedWithFaceSelection(ApplyInstantIDAdvanced):
    @classmethod
    def INPUT_TYPES(s):
        original_types = super().INPUT_TYPES()
        # Add face_selection as a new required input
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

        # Extract face embeddings
        face_embed = extractFeatures(insightface, image, face_selection=face_selection)
        if face_embed is None:
            raise Exception('Reference Image: No face detected.')

        # Extract face keypoints
        if image_kps is not None:
            face_kps = extractFeatures(insightface, image_kps, extract_kps=True, face_selection=face_selection)
        else:
            face_kps = extractFeatures(insightface, image[0].unsqueeze(0), extract_kps=True, face_selection=face_selection)

        if face_kps is None:
            face_kps = torch.zeros_like(image if image_kps is None else image_kps)
            print(f"\033[33mWARNING: No face detected in the keypoints image!\033[0m")

        # Process embeddings
        clip_embed = face_embed
        if clip_embed.shape[0] > 1:
            if combine_embeds == 'average':
                clip_embed = torch.mean(clip_embed, dim=0).unsqueeze(0)
            elif combine_embeds == 'norm average':
                clip_embed = torch.mean(clip_embed / torch.norm(clip_embed, dim=0, keepdim=True), dim=0).unsqueeze(0)

        # Apply noise if needed
        if noise > 0:
            seed = int(torch.sum(clip_embed).item()) % 1000000007
            torch.manual_seed(seed)
            clip_embed_zeroed = noise * torch.rand_like(clip_embed)
        else:
            clip_embed_zeroed = torch.zeros_like(clip_embed)

        # Initialize InstantID
        self.instantid = instantid
        self.instantid.to(self.device, dtype=self.dtype)

        # Get image embeddings
        image_prompt_embeds, uncond_image_prompt_embeds = self.instantid.get_image_embeds(
            clip_embed.to(self.device, dtype=self.dtype), 
            clip_embed_zeroed.to(self.device, dtype=self.dtype)
        )

        # Call parent implementation
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
            image_kps=face_kps,
            mask=mask,
            combine_embeds=combine_embeds
        )

# Update the global mappings
NODE_CLASS_MAPPINGS.update({
    "ApplyInstantIDAdvancedWithFaceSelection": ApplyInstantIDAdvancedWithFaceSelection,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "ApplyInstantIDAdvancedWithFaceSelection": "Apply InstantID Advanced with Face Selection",
})