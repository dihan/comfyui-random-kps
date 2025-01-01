import torch
import comfy.model_management
import torchvision.transforms as T
import numpy as np
import sys
import os
import traceback
from copy import deepcopy


# Add InstantID path to system path
INSTANTID_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ComfyUI_InstantID'))
if os.path.exists(INSTANTID_PATH):
    if INSTANTID_PATH not in sys.path:
        sys.path.append(INSTANTID_PATH)
        print(f"Added InstantID path: {INSTANTID_PATH}")
else:
    print(f"Warning: InstantID path not found at {INSTANTID_PATH}")

# Import InstantID components
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



def select_face(faces, method='largest'):
    """Select a face based on different criteria"""
    if not faces:
        return None
        
    faces_with_metrics = []
    for face in faces:
        bbox = face['bbox']
        area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        center_x = (bbox[0] + bbox[2])/2
        center_y = (bbox[1] + bbox[3])/2
        faces_with_metrics.append({
            'face': face,
            'area': area,
            'center_x': center_x,
            'center_y': center_y,
            'left': bbox[0],
            'right': bbox[2],
            'top': bbox[1],
            'bottom': bbox[3]
        })

    if method == 'largest':
        selected = max(faces_with_metrics, key=lambda x: x['area'])
    elif method == 'smallest':
        selected = min(faces_with_metrics, key=lambda x: x['area'])
    elif method == 'leftmost':
        selected = min(faces_with_metrics, key=lambda x: x['left'])
    elif method == 'rightmost':
        selected = max(faces_with_metrics, key=lambda x: x['right'])
    elif method == 'topmost':
        selected = min(faces_with_metrics, key=lambda x: x['top'])
    elif method == 'bottommost':
        selected = max(faces_with_metrics, key=lambda x: x['bottom'])
    elif method == 'center':
        selected = min(faces_with_metrics, 
                      key=lambda x: ((x['center_x']/640 - 0.5)**2 + 
                                   (x['center_y']/640 - 0.5)**2)**0.5)
    
    print(f"Selected face using '{method}' method:")
    print(f"- Area: {selected['area']:.2f}")
    print(f"- Position: ({selected['center_x']:.1f}, {selected['center_y']:.1f})")
    print(f"- Bounds: L={selected['left']:.1f}, R={selected['right']:.1f}, T={selected['top']:.1f}, B={selected['bottom']:.1f}")
    
    return selected['face']

class ModifiedInsightFace:
    """Simple wrapper that only modifies the face selection behavior"""
    def __init__(self, original_insightface, selection_method):
        self.original = original_insightface
        self.selection_method = selection_method
        # Keep the det_model reference for size adjustments
        self.det_model = self.original.det_model

    def get(self, img):
        """Override get method to modify face selection"""
        faces = self.original.get(img)
        if faces:
            return [select_face(faces, self.selection_method)]
        return None

    def __getattr__(self, attr):
        """Pass through all other attribute access to the original object"""
        return getattr(self.original, attr)

def select_face(faces, method='largest'):
    """Select a face based on different criteria"""
    if not faces:
        return None
        
    faces_with_metrics = []
    for face in faces:
        bbox = face['bbox']
        area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        center_x = (bbox[0] + bbox[2])/2
        center_y = (bbox[1] + bbox[3])/2
        faces_with_metrics.append({
            'face': face,
            'area': area,
            'center_x': center_x,
            'center_y': center_y,
            'left': bbox[0],
            'right': bbox[2],
            'top': bbox[1],
            'bottom': bbox[3]
        })

    if method == 'largest':
        selected = max(faces_with_metrics, key=lambda x: x['area'])
    elif method == 'smallest':
        selected = min(faces_with_metrics, key=lambda x: x['area'])
    elif method == 'leftmost':
        selected = min(faces_with_metrics, key=lambda x: x['left'])
    elif method == 'rightmost':
        selected = max(faces_with_metrics, key=lambda x: x['right'])
    elif method == 'topmost':
        selected = min(faces_with_metrics, key=lambda x: x['top'])
    elif method == 'bottommost':
        selected = max(faces_with_metrics, key=lambda x: x['bottom'])
    elif method == 'center':
        selected = min(faces_with_metrics, 
                      key=lambda x: ((x['center_x']/640 - 0.5)**2 + 
                                   (x['center_y']/640 - 0.5)**2)**0.5)
    
    print(f"Selected face using '{method}' method:")
    print(f"- Area: {selected['area']:.2f}")
    print(f"- Position: ({selected['center_x']:.1f}, {selected['center_y']:.1f})")
    print(f"- Bounds: L={selected['left']:.1f}, R={selected['right']:.1f}, T={selected['top']:.1f}, B={selected['bottom']:.1f}")
    
    return selected['face']

def extract_features_with_selection(insightface, image, extract_kps=False, selection_method='largest'):
    """Our custom feature extraction with face selection"""
    face_img = tensor_to_image(image)
    out = []

    insightface.det_model.input_size = (640,640)

    for i in range(face_img.shape[0]):
        for size in [(size, size) for size in range(640, 128, -64)]:
            insightface.det_model.input_size = size
            faces = insightface.get(face_img[i])
            if faces:
                selected_face = faces[0]  # Use the face already selected by ModifiedInsightFace
                
                if extract_kps:
                    kps_img = draw_kps(face_img[i], selected_face['kps'])
                    # Convert PIL Image to numpy array
                    kps_array = np.array(kps_img)
                    # Convert to tensor and normalize
                    kps_tensor = torch.from_numpy(kps_array).float() / 255.0
                    kps_tensor = kps_tensor.permute(2, 0, 1)  # Change from HWC to CHW
                    out.append(kps_tensor)
                else:
                    out.append(torch.from_numpy(selected_face['embedding']).unsqueeze(0))

                if 640 not in size:
                    print(f"\033[33mINFO: InsightFace detection resolution lowered to {size}.\033[0m")
                break

    if out:
        if extract_kps:
            out = torch.stack(out, dim=0)  # Stack tensors directly
        else:
            out = torch.stack(out, dim=0)
        return out
    return None

class InstantIDFace(ApplyInstantIDAdvanced):
    @classmethod
    def INPUT_TYPES(s):
        base_types = super().INPUT_TYPES()
        base_types['required']['face_selection_method'] = (
            ["largest", "smallest", "leftmost", "rightmost", "topmost", "bottommost", "center"],
            {"default": "largest"}
        )
        return base_types

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("MODEL", "positive", "negative",)
    FUNCTION = "apply_instantid"
    CATEGORY = "InstantID"

    def apply_instantid(self, instantid, insightface, control_net, image, model, positive, negative, 
                    ip_weight, cn_strength, start_at, end_at, noise,
                    image_kps=None, mask=None, combine_embeds='average', face_selection_method='largest'):
        try:
            print("\n=== Starting InstantID Face Processing ===")
            print(f"Face selection method: {face_selection_method}")
            
            dtype = comfy.model_management.unet_dtype()
            if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32
            
            self.dtype = dtype
            self.device = comfy.model_management.get_torch_device()

            # Create modified insightface that enforces our face selection
            modified_insightface = ModifiedInsightFace(insightface, face_selection_method)

            # Extract face features using selected method
            face_embed = extract_features_with_selection(modified_insightface, image, selection_method=face_selection_method)
            
            if face_embed is None:
                print("\033[33mWARNING: No faces detected in reference image. Processing will be limited.\033[0m")
                return super().apply_instantid(
                    instantid=instantid,
                    insightface=modified_insightface,
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

            # Extract keypoints using modified insightface
            face_kps = extract_features_with_selection(
                modified_insightface, 
                image_kps if image_kps is not None else image[0].unsqueeze(0), 
                extract_kps=True,
                selection_method=face_selection_method
            )

            if face_kps is None:
                face_kps = torch.zeros_like(image) if image_kps is None else image_kps
                print(f"\033[33mWARNING: No face detected in the keypoints image!\033[0m")

            # Modified embedding processing for better color preservation
            clip_embed = face_embed
            if clip_embed.shape[0] > 1:
                if combine_embeds == 'average':
                    # Normalize before averaging to preserve feature scales
                    norms = torch.norm(clip_embed, dim=-1, keepdim=True)
                    normalized_embeds = clip_embed / norms
                    clip_embed = torch.mean(normalized_embeds, dim=0).unsqueeze(0)
                    # Rescale to original magnitude
                    avg_norm = torch.mean(norms)
                    clip_embed = clip_embed * avg_norm
                elif combine_embeds == 'norm average':
                    clip_embed = torch.mean(clip_embed / torch.norm(clip_embed, dim=-1, keepdim=True), dim=0).unsqueeze(0)

            # Print embedding statistics for debugging
            print(f"Embedding stats before noise - Mean: {clip_embed.mean():.4f}, Std: {clip_embed.std():.4f}")

            if noise > 0:
                # Modified noise application
                seed = int(torch.sum(clip_embed).item()) % 1000000007
                torch.manual_seed(seed)
                
                # Generate noise that preserves embedding statistics
                noise_tensor = torch.randn_like(clip_embed)
                noise_tensor = noise_tensor * clip_embed.std() + clip_embed.mean()
                
                # Blend original embedding with noise
                clip_embed_zeroed = noise * noise_tensor + (1 - noise) * torch.zeros_like(clip_embed)
                
                print(f"Noise blend stats - Mean: {clip_embed_zeroed.mean():.4f}, Std: {clip_embed_zeroed.std():.4f}")
            else:
                clip_embed_zeroed = torch.zeros_like(clip_embed)

            self.instantid = instantid
            self.instantid.to(self.device, dtype=self.dtype)

            # Generate image embeddings
            image_prompt_embeds, uncond_image_prompt_embeds = self.instantid.get_image_embeds(
                clip_embed.to(self.device, dtype=self.dtype), 
                clip_embed_zeroed.to(self.device, dtype=self.dtype)
            )

            # Adjust strength parameters for better stability
            actual_ip_weight = min(ip_weight, 1.0) if ip_weight > 0 else ip_weight
            actual_cn_strength = min(cn_strength * 0.8, 1.0) if cn_strength > 0 else cn_strength

            return super().apply_instantid(
                instantid=instantid,
                insightface=modified_insightface,
                control_net=control_net,
                image=image,
                model=model,
                positive=positive,
                negative=negative,
                ip_weight=actual_ip_weight,
                cn_strength=actual_cn_strength,
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

NODE_DISPLAY_NAME_MAPPINGS = {
    "InstantIDFace": "InstantID Face",
}