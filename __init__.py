import torch
import numpy as np
import PIL.Image
import math
import cv2
import random

class RandomFaceKeypoints:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "point_size": ("INT", {"default": 10, "min": 1, "max": 100}),  # Increased max
                "line_width": ("INT", {"default": 4, "min": 1, "max": 100}),   # Increased max
                "min_scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1}),  # Increased max
                "max_scale": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1}), # Increased max
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "keypoints"

    def draw_kps(self, h, w, kps, point_size, line_width, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
        limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])

        out_img = np.zeros([h, w, 3])

        # Draw lines
        for i in range(len(limbSeq)):
            index = limbSeq[i]
            color = color_list[index[0]]

            x = kps[index][:, 0]
            y = kps[index][:, 1]
            length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
            polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), 
                                     (int(length / 2), line_width), 
                                     int(angle), 0, 360, 1)
            out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
        
        out_img = (out_img * 0.6).astype(np.uint8)

        # Draw points
        for idx_kp, kp in enumerate(kps):
            color = color_list[idx_kp]
            x, y = kp
            out_img = cv2.circle(out_img.copy(), (int(x), int(y)), point_size, color, -1)

        return out_img

    def generate_base_keypoints(self):
        """Generate the base keypoint structure"""
        return np.array([
            [-30, -20],  # Red point (left)
            [30, -20],   # Green point (right)
            [0, 0],      # Blue point (center)
            [0, -40],    # Yellow point (top)
            [0, 20]      # Purple point (bottom)
        ], dtype=np.float32)

    def transform_keypoints(self, kps, width, height, scale, tx, ty, rotation=0):
        """Apply transformation to keypoints"""
        # Scale
        kps = kps * scale

        # Rotation (in radians)
        cos_theta = math.cos(rotation)
        sin_theta = math.sin(rotation)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        kps = np.dot(kps, rotation_matrix.T)

        # Translation
        kps[:, 0] += tx
        kps[:, 1] += ty

        return kps

    def generate(self, latent, point_size=10, line_width=4, min_scale=0.5, max_scale=1.5, seed=0):
        # Get dimensions from latent
        batch_size = latent["samples"].shape[0]
        height = latent["samples"].shape[2] * 8  # Convert from latent to pixel space
        width = latent["samples"].shape[3] * 8   # Convert from latent to pixel space

        # Scale point size and line width based on image dimensions
        base_size = 512  # Reference size
        size_factor = min(width, height) / base_size
        scaled_point_size = int(point_size * size_factor)
        scaled_line_width = int(line_width * size_factor)

        # Initialize output tensor for batch
        output_tensors = []

        for b in range(batch_size):
            # Set seed per batch item
            batch_seed = seed + b
            random.seed(batch_seed)
            np.random.seed(batch_seed)

            # Generate base keypoints
            base_kps = self.generate_base_keypoints()

            # Scale base keypoints based on image size
            base_kps = base_kps * (size_factor * 1.5)  # 1.5 is a factor to make keypoints more visible

            # Random transformations
            scale = random.uniform(min_scale, max_scale)
            
            # Random position (with padding to keep face inside image)
            padding = scaled_point_size * 5  # Adjust padding based on point size
            tx = random.uniform(padding, width - padding)
            ty = random.uniform(padding, height - padding)

            # Random rotation
            rotation = random.uniform(-math.pi/6, math.pi/6)  # ±30 degrees

            # Apply transformations
            transformed_kps = self.transform_keypoints(
                base_kps, 
                width, 
                height, 
                scale,
                tx, 
                ty,
                rotation
            )

            # Generate the keypoint visualization
            out_img = self.draw_kps(height, width, transformed_kps, scaled_point_size, scaled_line_width)

            # Convert to tensor
            tensor = torch.from_numpy(out_img).float() / 255.0
            output_tensors.append(tensor)

        # Stack all batch items
        final_tensor = torch.stack(output_tensors, dim=0)

        return (final_tensor,)

NODE_CLASS_MAPPINGS = {
    "RandomFaceKeypoints": RandomFaceKeypoints
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomFaceKeypoints": "Random Face Keypoints"
}