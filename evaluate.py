import os
import torch
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt

from src.datasets.busi import BUSIDataset
from src.models.unet_baseline import UNet
from src.models.attention_unet import AttUNet
from src.metrics import dice_coef, iou_score


def get_model(name, in_ch):
    if name == "unet":
        return UNet(in_ch=in_ch, out_ch=1, base=32)
    elif name == "attunet":
        return AttUNet(in_ch=in_ch, out_ch=1, base=32)
    else:
        raise ValueError("Unknown model type")


def run_evaluate(args):
    device = torch.device("cpu")

    # Dataset
    ds = BUSIDataset(
        args.data_root,
        image_size=args.image_size,
        use_gaze=args.use_gaze
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

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

    # Evaluation loop
    val_loss, dices, ious = 0.0, [], []
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in loader:
            if args.use_gaze:
                imgs, masks, gazes = batch
                imgs = torch.cat([imgs, gazes], dim=1)
            else:
                imgs, masks = batch

            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = criterion(logits, masks)

            val_loss += loss.item()
            dices.append(dice_coef(logits, masks))
            ious.append(iou_score(logits, masks))

    avg_loss = val_loss / max(1, len(loader))
    avg_dice = sum(dices) / max(1, len(dices))
    avg_iou = sum(ious) / max(1, len(ious))

    print(f"✅ Model: {args.model_type} | Gaze: {args.use_gaze}")
    print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  Avg Dice: {avg_dice:.4f}")
    print(f"  Avg IoU : {avg_iou:.4f}")

    # Plot first sample as visual check
    imgs, masks = next(iter(loader))[:2]
    if args.use_gaze:
        imgs, masks, gazes = next(iter(loader))
        imgs = torch.cat([imgs, gazes], dim=1)

    preds = torch.sigmoid(model(imgs.to(device)))
    pred_mask = (preds[0, 0] > 0.5).cpu().numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    ax1.imshow(imgs[0, 0], cmap="gray")
    ax1.set_title("Input")
    ax2.imshow(masks[0, 0], cmap="gray")
    ax2.set_title("Ground Truth")
    ax3.imshow(pred_mask, cmap="gray")
    ax3.set_title("Prediction")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="Dataset_BUSI_with_GT")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--model_type", type=str, choices=["unet", "attunet"], required=True)
    parser.add_argument("--use_gaze", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="saved_models")
    parser.add_argument("--weights_path", type=str, default=None)
    args = parser.parse_args()

    run_evaluate(args)
