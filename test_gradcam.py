"""
test_gradcam.py

Quick test script for Grad-CAM proxy gaze maps.
Loads a trained model and generates a heatmap for a dummy image.
"""

import torch
from src.models.attention_unet import AttUNet
from src.grad_cam_proxy import GradCAM


def main():
    # Load trained Attention U-Net
    model = AttUNet(in_ch=1, out_ch=1, base=32)
    model.load_state_dict(torch.load("saved_models/attunet_cpu.pth", map_location="cpu"))
    model.eval()

    # Pick a layer for Grad-CAM
    # In your implementation, inc is already a DoubleConv
    target_layer = model.inc

    # Initialize GradCAM
    gradcam = GradCAM(model, target_layer)

    # Test with a dummy input image (128x128 grayscale)
    img = torch.randn(1, 1, 128, 128)
    heatmap = gradcam.generate(img)

    print("✅ Grad-CAM proxy heatmap generated.")
    print("Shape:", heatmap.shape)  # (1, 128, 128)


if __name__ == "__main__":
    main()
