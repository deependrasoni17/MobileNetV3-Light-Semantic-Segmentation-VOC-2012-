import os
# CRITICAL FIX 1: OpenMP environment variables must be set before any other imports
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import random
import time
import copy
from glob import glob
from argparse import ArgumentParser

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import your custom architecture
from model import MobileNetV3LightSeg

# ==========================================
# 1. UTILS & DATASET
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# Import data handling code from the dataset module to avoid duplication
from dataset import (
    VOCFolderSegDataset, get_transforms, list_image_ids, create_and_save_split
)

# ==========================================
# 2. LOSS FUNCTIONS & METRICS
# ==========================================
class DiceLossPerClass(nn.Module):
    def __init__(self, num_classes=21, ignore_index=255, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits, target):
        B, C, H, W = logits.shape
        mask_valid = (target != self.ignore_index)
        probs = torch.softmax(logits, dim=1)

        target_onehot = torch.zeros_like(probs)
        valid_idx = mask_valid.unsqueeze(1).expand(-1, C, -1, -1)

        t = target.clone()
        t[~mask_valid] = 0
        target_onehot.scatter_(1, t.unsqueeze(1), 1.0)
        target_onehot = target_onehot * valid_idx.float()

        dims = (0, 2, 3)
        intersection = (probs * target_onehot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + target_onehot.sum(dim=dims)
        dice_score = (2.0 * intersection + self.eps) / (cardinality + self.eps)

        class_present = (target_onehot.sum(dim=dims) > 0).float()
        if class_present.sum() > 0:
            dice_mean = (dice_score * class_present).sum() / (class_present.sum())
        else:
            dice_mean = dice_score.mean()

        return 1.0 - dice_mean, dice_score

def compute_class_weights(dataset, num_classes=21, loader_bs=8, num_workers=0):
    counts = np.zeros(num_classes, dtype=np.float64)
    loader = DataLoader(dataset, batch_size=loader_bs, shuffle=False, num_workers=num_workers)
    for imgs, masks in loader:
        m = masks.cpu().numpy()
        for c in range(num_classes):
            counts[c] += np.sum(m == c)
    counts = np.maximum(counts, 1.0)
    freq = counts / counts.sum()
    inv_freq = 1.0 / freq
    weights = inv_freq / np.mean(inv_freq)
    return torch.from_numpy(weights.astype(np.float32))

def get_batch_inter_union(preds, masks, num_classes=21, ignore_index=255):
    with torch.inference_mode():
        inter = np.zeros(num_classes, dtype=np.float64)
        union = np.zeros(num_classes, dtype=np.float64)
        valid_mask = (masks != ignore_index)

        for c in range(num_classes):
            pred_c = (preds == c) & valid_mask
            true_c = (masks == c) & valid_mask
            inter[c] = (pred_c & true_c).sum().item()
            union[c] = pred_c.sum().item() + true_c.sum().item()
    return inter, union

# ==========================================
# 3. TRAINING & VALIDATION LOOPS
# ==========================================
def train_one_epoch(model, dataloader, optimizer, ce_loss_fn, dice_loss_fn, device, ignore_index=255, scaler=None):
    model.train()
    running_loss = 0.0
    n_samples = 0
    for imgs, masks in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad()

        if scaler is not None:
            # Modern PyTorch AMP syntax
            with torch.amp.autocast(device_type=device.type):
                logits = model(imgs)
                ce = ce_loss_fn(logits, masks)
                dice_loss, _ = dice_loss_fn(logits, masks)
                loss = ce + 1.0 * dice_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            ce = ce_loss_fn(logits, masks)
            dice_loss, _ = dice_loss_fn(logits, masks)
            loss = ce + 1.0 * dice_loss
            loss.backward()
            optimizer.step()

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        n_samples += bs
    return running_loss / max(1, n_samples)

def validate(model, dataloader, device, num_classes=21):
    model.eval()
    total_inter = np.zeros(num_classes, dtype=np.float64)
    total_union = np.zeros(num_classes, dtype=np.float64)

    with torch.inference_mode():
        for imgs, masks in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            preds = model(imgs) 
            batch_inter, batch_union = get_batch_inter_union(preds, masks, num_classes)
            total_inter += batch_inter
            total_union += batch_union

    per_class_dice = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if total_union[c] > 0:
            per_class_dice[c] = (2.0 * total_inter[c]) / total_union[c]
        else:
            per_class_dice[c] = 1.0

    avg_macro = per_class_dice.mean()
    return avg_macro, per_class_dice

# ==========================================
# 4. MAIN EXECUTOR
# ==========================================
def main(args):
    set_seed(args.seed)
    if args.use_dml:
        import torch_directml
        device = torch_directml.device()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    ensure_dir(args.output_dir)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb"))
    
    try:
        train_transform, val_transform = get_transforms(
            img_size=args.img_size, corruption_prob=args.corruption_prob)

        jpeg_dir = os.path.join(args.voc_root, "JPEGImages")
        seg_dir = os.path.join(args.voc_root, "SegmentationClass")
        
        if os.path.exists(args.splits_json):
            print(f"Loading existing splits from {args.splits_json}")
            with open(args.splits_json, "r") as f:
                splits = json.load(f)
            train_ids = splits["train_ids"]
            val_ids = splits["val_ids"]
        else:
            print("Creating new train/val splits...")
            all_image_ids = list_image_ids(jpeg_dir)
            filtered_ids = [img_id for img_id in all_image_ids if os.path.exists(os.path.join(seg_dir, img_id + ".png"))]
            train_ids, val_ids = create_and_save_split(filtered_ids, args.train_frac, args.seed, args.splits_json)

        train_dataset = VOCFolderSegDataset(voc_root=args.voc_root, image_ids=train_ids, transform=train_transform)
        val_dataset = VOCFolderSegDataset(voc_root=args.voc_root, image_ids=val_ids, transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

        print("Computing class weights from training set...")
        class_weights = compute_class_weights(train_dataset, num_classes=args.num_classes,
                                              loader_bs=args.loader_bs, num_workers=args.num_workers)
        class_weights = class_weights.to(device)

        model = MobileNetV3LightSeg(num_classes=args.num_classes, pretrained=args.pretrained)
        model = model.to(device)

        ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=args.ignore_index)
        dice_loss_fn = DiceLossPerClass(num_classes=args.num_classes, ignore_index=args.ignore_index)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        # Enable AMP for both CUDA and DirectML (privateuseone)
        scaler = torch.amp.GradScaler() if args.use_amp and device.type in ["cuda", "privateuseone"] else None

        start_epoch = 0
        best_val = -1.0
        patience_counter = 0

        if args.resume and os.path.exists(args.resume):
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optim_state"])
            scheduler.load_state_dict(ckpt.get("scheduler_state", scheduler.state_dict()))
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val = ckpt.get("best_val", best_val)
            print(f"Resumed from {args.resume} at epoch {start_epoch}, best_val={best_val}")

        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()

            train_loss = train_one_epoch(model, train_loader, optimizer, ce_loss_fn, dice_loss_fn, device, args.ignore_index, scaler)
            val_macro, per_class = validate(model, val_loader, device, num_classes=args.num_classes)
            scheduler.step()

            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/macro_dice", val_macro, epoch)
            for c in range(args.num_classes):
                writer.add_scalar(f"val/dice_class_{c}", float(per_class[c]), epoch)

            print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} | val_macro_dice {val_macro:.4f} | time {(time.time()-t0):.1f}s")

            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val": best_val
            }
            
            torch.save(ckpt, os.path.join(args.output_dir, "last_checkpoint.pth"))

            if val_macro > best_val:
                best_val = val_macro
                ckpt["best_val"] = best_val
                torch.save(ckpt, os.path.join(args.output_dir, "best_checkpoint.pth"))
                patience_counter = 0
                print(f"  New best model saved with val_macro_dice={best_val:.4f}")
            else:
                patience_counter += 1
                print(f"  No improvement. patience {patience_counter}/{args.patience}")

            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break
    finally:
        writer.close()
    return best_val

def run_hyperparameter_tuning(base_args):
    # Adjust arrays below to iterate through multiple values
    learning_rates = [5e-4, 1e-3, 5e-3]
    batch_sizes = [16, 8, 32]

    best_overall_dice = -1.0
    best_params = {}

    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"\n{'='*50}\nSTARTING TRIAL: Learning Rate = {lr}, Batch Size = {bs}\n{'='*50}")
            trial_args = copy.deepcopy(base_args)
            trial_args.lr = lr
            trial_args.batch_size = bs
            trial_args.output_dir = f"./checkpoints/tune_lr_{lr}_bs_{bs}"
            
            trial_best_val = main(trial_args)

            if trial_best_val > best_overall_dice:
                best_overall_dice = trial_best_val
                best_params = {'lr': lr, 'batch_size': bs}

    print(f"\n{'*'*50}\nHYPERPARAMETER TUNING COMPLETE!\nBest Macro-DSC: {best_overall_dice:.4f}\nWinning Hyperparameters: {best_params}\n{'*'*50}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--voc_root", type=str, default="C:/Users/ARKAJYOTI DAS/Downloads/mini_competition_group_15/archive (11)/VOC2012_train_val/VOC2012_train_val")
    parser.add_argument("--splits_json", type=str, default="C:/Users/ARKAJYOTI DAS/Downloads/mini_competition_group_15/archive (11)/voc2012_splits.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--loader_bs", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader. Default 0 is safest for Windows.")
    parser.add_argument("--img_size", type=int, default=300)
    parser.add_argument("--corruption_prob", type=float, default=0.3)
    parser.add_argument("--num_classes", type=int, default=21)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--use_dml", action="store_true", help="Use DirectML for Intel GPU on Windows")

    args = parser.parse_args()
    
    # Run the tuning wrapper (or just standard training if arrays have length 1)
    run_hyperparameter_tuning(args)