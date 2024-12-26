## Installation

1. Navigate to your ComfyUI custom nodes directory
2. Clone this repository:
   ```bash
   cd custom_nodes
   git clone https://github.com/dihan/comfyui-random-kps


# Random KPS Generator and InstantIDFace Extensions

This repository contains two complementary components that enhance face generation workflows in ComfyUI when used with InstantID:

## Random KPS Generator

A utility node that generates random facial keypoint structures that can be connected to cubiq's InstantID workflow. 

### Features:
- Generates facial keypoint visualization with 5 key points:
  - Left eye (Red)
  - Right eye (Green)
  - Nose (Blue)
  - Right mouth corner (Purple)
  - Left mouth corner (Yellow)
- Can be used as input for InstantID face generation
- Supports random variations in:
  - Position
  - Scale
  - Rotation
- Compatible with cubiq's ComfyUI_InstantID workflow

## InstantIDFace

A modified version of the InstantID node from cubiq's ComfyUI_InstantID that adds the ability to select smaller faces when multiple faces are detected in an image.

### Key Features:
- Adds face selection mode:
  - "largest" (default behavior from original InstantID)
  - "smallest" (new option to select the smallest detected face)
- Maintains all original InstantID functionality
- Useful for images containing multiple faces where you want to focus on the smaller face

### Dependencies:
- Requires base ComfyUI installation
- Requires cubiq's ComfyUI_InstantID

### Usage Notes:
These components can be used independently or together within your ComfyUI workflow. The Random KPS Generator provides alternative keypoint inputs for InstantID, while InstantIDFace gives you more control over which face is selected for reference in multi-face images.
