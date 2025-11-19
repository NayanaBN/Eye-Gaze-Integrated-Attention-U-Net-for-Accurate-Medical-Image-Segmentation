"""
grad_cam_proxy.py

Hybrid Grad-CAM proxy:
- For training: returns a dummy Gaussian heatmap (fast).
- For evaluation/visualization: can generate real Grad-CAM.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook for activations
        def forward_hook(module, input, output):
            self.activations = output.detach()

        # Hook for gradients
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate(self, x, use_grad=True):
        """
        Generate Grad-CAM or dummy gaze map.

        Args:
            x (torch.Tensor): input tensor [B, C, H, W]
            use_grad (bool): if False → return dummy Gaussian heatmap

        Returns:
            torch.Tensor: heatmap [1, H, W]
        """
        h, w = x.shape[2:]

        if not use_grad:
            # Fast dummy Gaussian heatmap
            heatmap = np.zeros((h, w), dtype=np.float32)
            cv2.circle(heatmap, (w // 2, h // 2), min(h, w) // 4, 1, -1)
            heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
            return torch.tensor(heatmap).unsqueeze(0)

        # --- Real Grad-CAM ---
        x = x.requires_grad_(True)
        logits = self.model(x)
        target = logits[:, 0, :, :].mean()
        self.model.zero_grad()
        target.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP on gradients
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (w, h))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)

        return torch.tensor(cam, dtype=torch.float32).unsqueeze(0)
