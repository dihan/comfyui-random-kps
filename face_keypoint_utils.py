"""Utility functions and classes for face keypoint generation"""
import torch
import numpy as np
import cv2
import math
import random

class FaceKeypointGenerator:
    def draw_kps(self, h, w, kps, point_size, line_width, color_list=[(255,0,0), (0,255,0), (0,0,255), (128,0,128), (255,255,0)]):
        limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])  # All points connect to nose (index 2)

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

        # Draw points in correct order with correct colors
        for idx_kp, kp in enumerate(kps):
            color = color_list[idx_kp]
            x, y = kp
            out_img = cv2.circle(out_img.copy(), (int(x), int(y)), point_size, color, -1)

        return out_img

    def generate_base_keypoints(self, size_factor=1.0):
        """Generate default base keypoint structure in correct facial order"""
        return np.array([
            [-30, -20],   # Red point (left eye)
            [30, -20],    # Green point (right eye)
            [0, 0],       # Blue point (nose)
            [20, 20],     # Purple point (right mouth)
            [-20, 20],    # Yellow point (left mouth)
        ], dtype=np.float32) * size_factor

    def extract_keypoints_from_image(self, image):
        """Extract keypoint positions from reference image in correct order"""
        if isinstance(image, torch.Tensor):
            image = image.squeeze(0).cpu().numpy()
            if len(image.shape) == 4:
                image = image[0]
        
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)

        # Define color thresholds for each point
        color_thresholds = {
            'red': ([200, 0, 0], [255, 50, 50]),       # Left eye
            'green': ([0, 200, 0], [50, 255, 50]),     # Right eye
            'blue': ([0, 0, 200], [50, 50, 255]),      # Nose
            'purple': ([128, 0, 128], [255, 0, 255]),  # Right mouth
            'yellow': ([200, 200, 0], [255, 255, 50])  # Left mouth
        }

        # Extract points in specific order
        keypoints = []
        color_order = ['red', 'green', 'blue', 'purple', 'yellow']
        
        for color in color_order:
            lower, upper = color_thresholds[color]
            mask = cv2.inRange(image, np.array(lower), np.array(upper))
            points = np.argwhere(mask > 0)
            if len(points) > 0:
                center = np.mean(points, axis=0)
                keypoints.append(center)

        if len(keypoints) == 5:
            return np.array(keypoints)
        return None

    def transform_keypoints(self, kps, width, height, scale, tx, ty, rotation=0):
        """Apply transformation to keypoints"""
        # Get center of the keypoints
        center = np.mean(kps, axis=0)
        
        # Center the keypoints
        centered_kps = kps - center

        # Scale
        scaled_kps = centered_kps * scale

        # Rotation (in radians)
        cos_theta = math.cos(rotation)
        sin_theta = math.sin(rotation)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        rotated_kps = np.dot(scaled_kps, rotation_matrix.T)

        # Translation to new position
        transformed_kps = rotated_kps + np.array([tx, ty])

        return transformed_kps

    def generate_keypoints(self, latent, point_size, line_width, min_scale, max_scale, seed, reference_kps=None):
        # Get dimensions from latent
        batch_size = latent["samples"].shape[0]
        height = latent["samples"].shape[2] * 8
        width = latent["samples"].shape[3] * 8

        # Scale factors
        base_size = 512
        size_factor = min(width, height) / base_size
        scaled_point_size = int(point_size * size_factor)
        scaled_line_width = int(line_width * size_factor)

        # Get base keypoint structure
        if reference_kps is not None:
            base_kps = self.extract_keypoints_from_image(reference_kps)
            if base_kps is None:
                print("Warning: Could not extract keypoints from reference image. Using default structure.")
                base_kps = self.generate_base_keypoints(size_factor * 1.5)
            else:
                # Center the extracted keypoints around origin
                center = np.mean(base_kps, axis=0)
                base_kps = base_kps - center
        else:
            base_kps = self.generate_base_keypoints(size_factor * 1.5)

        output_tensors = []

        for b in range(batch_size):
            batch_seed = seed + b
            random.seed(batch_seed)
            np.random.seed(batch_seed)

            # Random transformations
            scale = random.uniform(min_scale, max_scale)
            
            # Random position (with padding)
            padding = scaled_point_size * 5
            tx = random.uniform(padding, width - padding)
            ty = random.uniform(padding, height - padding)

            # Random rotation
            rotation = random.uniform(-math.pi/6, math.pi/6)

            # Apply transformations
            transformed_kps = self.transform_keypoints(
                base_kps.copy(), 
                width, 
                height, 
                scale,
                tx, 
                ty,
                rotation
            )

            # Generate visualization
            out_img = self.draw_kps(height, width, transformed_kps, scaled_point_size, scaled_line_width)

            # Convert to tensor
            tensor = torch.from_numpy(out_img).float() / 255.0
            output_tensors.append(tensor)

        final_tensor = torch.stack(output_tensors, dim=0)
        return final_tensor