"""Node implementations for Face Keypoints"""
from .face_keypoint_utils import FaceKeypointGenerator

class RandomFaceKeypoints:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "point_size": ("INT", {"default": 10, "min": 1, "max": 100}),
                "line_width": ("INT", {"default": 4, "min": 1, "max": 100}),
                "min_scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1}),
                "max_scale": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            },
            "optional": {
                "reference_kps": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "keypoints"

    def __init__(self):
        self.generator = FaceKeypointGenerator()

    def generate(self, latent, point_size=10, line_width=4, min_scale=0.5, max_scale=1.5, seed=0, reference_kps=None):
        return (self.generator.generate_keypoints(latent, point_size, line_width, min_scale, max_scale, seed, reference_kps),)

NODE_CLASS_MAPPINGS = {
    "RandomFaceKeypoints": RandomFaceKeypoints
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomFaceKeypoints": "Random Face Keypoints"
}