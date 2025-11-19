import os
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse

from src.datasets.busi import BUSIDataset
from src.models.unet_baseline import UNet
from src.models.attention_unet import AttUNet
from src.metrics import dice_coef
from src.losses import HybridLoss


def get_model(name, in_ch):
    if name == "unet":
        return UNet(in_ch=in_ch, out_ch=1, base=32)
    elif name == "attunet":
        return AttUNet(in_ch=in_ch, out_ch=1, base=32)
    else:
        raise ValueError("Unknown model type")


def run_train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cpu")

    # Dataset
    ds = BUSIDataset(
        args.data_root,
        image_size=args.image_size,
        use_gaze=args.use_gaze,
        model_weights=None  # we don’t load pretrained for dataset gaze maps
    )
    val_size = int(len(ds) * args.val_split)
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # Model
    in_ch = 2 if args.use_gaze else 1
    model = get_model(args.model_type, in_ch).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = HybridLoss(lambda_bce=0.5)

    # Auto filename
    suffix = "gaze" if args.use_gaze else "baseline"
    save_name = f"{args.model_type}_{suffix}_cpu.pth"
    save_path = os.path.join(args.save_dir, save_name)

    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        # ------------------ TRAIN ------------------
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        for batch in pbar:
            if args.use_gaze:
                imgs, masks, gazes = batch
                imgs = torch.cat([imgs, gazes], dim=1)
            else:
                imgs, masks = batch

            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ------------------ VALIDATE ------------------
        model.eval()
        val_loss = 0.0
        dices = []
        with torch.no_grad():
            for batch in val_loader:
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

        val_loss /= max(1, len(val_loader))
        val_dice = sum(dices) / max(1, len(dices))
        print(f"Epoch {epoch}: val_loss={val_loss:.4f} | val_dice={val_dice:.4f}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), save_path)
            print(f"  ↳ Saved best model to {save_path} (dice={best_dice:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="Dataset_BUSI_with_GT")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--model_type", type=str, choices=["unet", "attunet"], required=True)
    parser.add_argument("--use_gaze", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--save_dir", type=str, default="saved_models")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    run_train(args)
