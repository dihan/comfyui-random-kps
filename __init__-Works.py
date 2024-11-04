import torch
import numpy as np
import PIL.Image
import math
import cv2

class RandomFaceKeypoints:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "point_size": ("INT", {"default": 10, "min": 1, "max": 50}),
                "line_width": ("INT", {"default": 4, "min": 1, "max": 20})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "keypoints"

    def draw_kps(self, h, w, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
        stickwidth = 4
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
                                     (int(length / 2), stickwidth), 
                                     int(angle), 0, 360, 1)
            out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
        
        out_img = (out_img * 0.6).astype(np.uint8)

        # Draw points
        for idx_kp, kp in enumerate(kps):
            color = color_list[idx_kp]
            x, y = kp
            out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

        return out_img

    def generate(self, width=512, height=512, point_size=10, line_width=4):
        # Calculate center point
        center_x = width // 2
        center_y = height // 2
        
        # Define the 5 keypoints around the center
        kps = np.array([
            [center_x - 30, center_y - 20],  # Red point (left)
            [center_x + 30, center_y - 20],  # Green point (right)
            [center_x, center_y],            # Blue point (center)
            [center_x, center_y - 40],       # Yellow point (top)
            [center_x, center_y + 20]        # Purple point (bottom)
        ])

        # Generate the keypoint visualization
        out_img = self.draw_kps(height, width, kps)

        # Convert to tensor format expected by ComfyUI
        tensor = torch.from_numpy(out_img).float() / 255.0
        
        # Add batch dimension if needed
        tensor = tensor.unsqueeze(0)

        return (tensor,)

NODE_CLASS_MAPPINGS = {
    "RandomFaceKeypoints": RandomFaceKeypoints
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomFaceKeypoints": "Random Face Keypoints"
}