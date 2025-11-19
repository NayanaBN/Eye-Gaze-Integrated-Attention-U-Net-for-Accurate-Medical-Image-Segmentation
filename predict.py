import os
import torch
import cv2
import argparse
import matplotlib.pyplot as plt

from src.models.unet_baseline import UNet
from src.models.attention_unet import AttUNet


def get_model(name, in_ch):
    if name == "unet":
        return UNet(in_ch=in_ch, out_ch=1, base=32)
    elif name == "attunet":
        return AttUNet(in_ch=in_ch, out_ch=1, base=32)
    else:
        raise ValueError("Unknown model type")


def run_predict(args):
    device = torch.device("cpu")

    # Model
    in_ch = 2 if args.use_gaze else 1
    model = get_model(args.model_type, in_ch).to(device)

    # Auto filename if not provided
    if args.weights_path is None:
        suffix = "gaze" if args.use_gaze else "baseline"
        auto_name = f"{args.model_type}_{suffix}_cpu.pth"
        args.weights_path = os.path.join(args.save_dir, auto_name)

    # Load checkpoint
    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.weights_path}")

    state = torch.load(args.weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Load and preprocess image
    img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    orig_shape = img.shape
    img_resized = cv2.resize(img, (args.image_size, args.image_size))
    img_norm = img_resized / 255.0
    img_tensor = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # For gaze-guided model, add dummy gaze channel
    if args.use_gaze:
        dummy_gaze = torch.zeros_like(img_tensor)
        img_tensor = torch.cat([img_tensor, dummy_gaze], dim=1)

    # Predict
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)
        pred_mask = (probs[0, 0] > 0.5).cpu().numpy().astype("uint8")

    # Resize back to original image size
    pred_resized = cv2.resize(pred_mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)

    # Save
    out_path = args.output_path
    cv2.imwrite(out_path, pred_resized * 255)

    print(f"✅ Prediction saved at {out_path}")

    # Show
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_resized, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["unet", "attunet"], required=True)
    parser.add_argument("--use_gaze", action="store_true")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--save_dir", type=str, default="saved_models")
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="predicted_mask.png")
    args = parser.parse_args()

    run_predict(args)
