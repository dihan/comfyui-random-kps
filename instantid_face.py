"""InstantID Face implementation with face selection and fixed keypoint extraction"""
import torch
import numpy as np
import comfy.model_management
import torchvision.transforms as T
import sys
import os
import cv2

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
    """Enhanced extractFeatures with improved batch handling and face detection"""
    face_img = tensor_to_image(image)
    out = []

    print(f"\nFace selection mode: {face_selection}")

    for i in range(face_img.shape[0]):
        print(f"\nProcessing batch image {i+1}/{face_img.shape[0]}")
        all_faces = []
        best_detection = None
        
        # Try all detection sizes for each image
        for size in [(640, 640), (512, 512), (384, 384)]:
            try:
                # Resize image to current detection size while maintaining aspect ratio
                current_img = face_img[i].copy()
                h, w = current_img.shape[:2]
                scale = min(size[0]/w, size[1]/h)
                new_w, new_h = int(w*scale), int(h*scale)
                resized_img = cv2.resize(current_img, (new_w, new_h))
                
                # Detect faces at current size
                faces = insightface.get(resized_img)
                
                if faces and len(faces) > 0:
                    print(f"Found {len(faces)} faces at size {size}")
                    
                    # Scale bbox and keypoints back to original size
                    for face in faces:
                        # Scale bbox back to original size
                        face['bbox'] = [
                            face['bbox'][0]/scale,
                            face['bbox'][1]/scale,
                            face['bbox'][2]/scale,
                            face['bbox'][3]/scale
                        ]
                        
                        # Scale keypoints back to original size
                        if 'kps' in face:
                            face['kps'] = face['kps'] / scale
                        
                        bbox = face['bbox']
                        area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
                        width = bbox[2]-bbox[0]
                        height = bbox[3]-bbox[1]
                        
                        if width > 20 and height > 20:  # Basic size validation
                            all_faces.append((area, face))
                            print(f"Valid face found - Area: {area:.2f}, Bbox: {bbox}")
                
            except Exception as e:
                print(f"Error during face detection at size {size}: {str(e)}")
                continue
        
        if all_faces:
            # Sort faces by area
            all_faces.sort(key=lambda x: x[0], reverse=True)
            print(f"Found {len(all_faces)} valid faces in total")
            
            # Select face based on mode
            if face_selection == "smallest" and len(all_faces) > 1:
                selected_area, selected_face = all_faces[-1]
                print(f"Selected smallest face (area: {selected_area:.2f})")
            else:
                selected_area, selected_face = all_faces[0]
                print(f"Selected largest face (area: {selected_area:.2f})")
            
            # Process selected face
            if extract_kps:
                try:
                    kps_img = draw_kps(face_img[i], selected_face['kps'])
                    batch_out = T.ToTensor()(kps_img).permute(1, 2, 0)
                    print("Generated keypoint visualization")
                except Exception as e:
                    print(f"Error generating keypoints: {str(e)}")
                    batch_out = torch.zeros_like(image[0])
            else:
                try:
                    batch_out = torch.from_numpy(selected_face['embedding']).unsqueeze(0)
                    print("Extracted face embedding")
                except Exception as e:
                    print(f"Error extracting embedding: {str(e)}")
                    continue
            
            out.append(batch_out)
        else:
            print(f"No valid faces detected in batch image {i+1}")
            if extract_kps:
                out.append(torch.zeros_like(image[0]))
            else:
                print("Skipping batch image due to no face detection")

    if len(out) > 0:
        try:
            return torch.stack(out, dim=0)
        except Exception as e:
            print(f"Error stacking outputs: {str(e)}")
            if extract_kps:
                return torch.zeros_like(image)
            return None
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


    def apply_instantid(self, instantid, insightface, control_net, image, model, positive, negative, 
                        ip_weight, cn_strength, start_at, end_at, noise, face_selection,
                        image_kps=None, mask=None, combine_embeds='average'):
        
        dtype = comfy.model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32
        
        self.dtype = dtype
        self.device = comfy.model_management.get_torch_device()

        # Extract face features with consistent face selection
        face_embed, selected_faces = extractFeatures(insightface, image, face_selection=face_selection)
        if face_embed is None or all(face is None for face in selected_faces):
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
        kps_image = image_kps if image_kps is not None else image[0].unsqueeze(0)
        face_kps, _ = extractFeatures(insightface, kps_image, extract_kps=True, face_selection=face_selection)

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
            image_kps=face_kps,
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