import os
import json
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from model import MobileNetV3LightSeg
from dataset import VOCFolderSegDataset

try:
    from thop import profile
except ImportError:
    print("WARNING: 'thop' library not found. Please run 'pip install thop' to calculate FLOPs.")
    profile = None

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==========================================
# 1. EVAL TRANSFORMS
# ==========================================
def get_eval_transforms(img_size=300, corrupted=False):
    if not corrupted:
        return A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

# ==========================================
# 2. EVALUATION LOGIC
# ==========================================
def run_evaluation(model, dataloader, device):
    model.eval()
    total_inter = 0.0
    total_union = 0.0

    print("Running evaluation loop...")
    with torch.inference_mode():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            # model in eval mode returns (B, H, W) integer mask directly
            preds = model(imgs)
            valid_mask = (masks != 255)
            pred_fg = (preds > 0) & valid_mask
            true_fg = (masks > 0) & valid_mask
            total_inter += (pred_fg & true_fg).sum().item()
            total_union += pred_fg.sum().item() + true_fg.sum().item()

    binary_dice = (2.0 * total_inter) / total_union if total_union > 0 else 1.0
    return binary_dice

# ==========================================
# 3. COMPUTATIONAL EFFICIENCY (FLOPs)
# ==========================================
def measure_efficiency(model, dataloader, device):
    if profile is None:
        return None, None

    model.eval()
    imgs, _ = next(iter(dataloader))
    real_input = imgs[0:1].to(device)

    print(f"\nCalculating FLOPs for input shape {real_input.shape}...")
    with torch.inference_mode():
        flops, params = profile(model, inputs=(real_input,), verbose=False)
    return flops, params

# ==========================================
# 4. MAIN SCRIPT
# ==========================================
def main(args):
    set_seed(42)
    if args.use_dml:
        import torch_directml
        device = torch_directml.device()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sanitize paths to prevent path traversal
    voc_root = os.path.realpath(args.voc_root)
    splits_json = os.path.realpath(args.splits_json)
    checkpoint_path = os.path.realpath(args.checkpoint_path)

    if not os.path.exists(splits_json):
        raise FileNotFoundError(f"Splits file not found at {splits_json}. Run prepare_data.py first.")

    with open(splits_json, "r") as f:
        val_ids = json.load(f)["val_ids"]

    print(f"Loading best model from {checkpoint_path}...")
    model = MobileNetV3LightSeg(num_classes=args.num_classes)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)

    # --- TEST 1: CLEAN ---
    print("\n--- TEST 1: CLEAN VALIDATION ---")
    safe_voc_root = os.path.realpath(voc_root)
    clean_ds = VOCFolderSegDataset(safe_voc_root, val_ids, transform=get_eval_transforms(img_size=args.img_size, corrupted=False))
    clean_loader = DataLoader(clean_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    clean_dice = run_evaluation(model, clean_loader, device)

    # --- TEST 2: CORRUPTED ---
    print("\n--- TEST 2: CORRUPTED VALIDATION ---")
    corr_ds = VOCFolderSegDataset(safe_voc_root, val_ids, transform=get_eval_transforms(img_size=args.img_size, corrupted=True))
    corr_loader = DataLoader(corr_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    corr_dice = run_evaluation(model, corr_loader, device)

    # --- TEST 3: EFFICIENCY ---
    flops, params = measure_efficiency(model, clean_loader, device)

    # --- FINAL SUMMARY ---
    print("\n=======================================================")
    print(" FINAL EVALUATION SUMMARY (ACCURACY & EFFICIENCY)")
    print("=======================================================")
    print(f" Binary Dice (Clean):     {clean_dice:.4f}")
    print(f" Binary Dice (Corrupted): {corr_dice:.4f}")
    if flops is not None:
        print(f" Computational FLOPs:     {flops / 1e9:.4f} GFLOPs")
        print(f" Model Parameters:        {params / 1e6:.2f} M")
    else:
        print(" Computational FLOPs:     [Requires 'thop' library]")
    print("=======================================================")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--voc_root", type=str, default="C:/Users/ARKAJYOTI DAS/Downloads/mini_competition_group_15/archive (11)/VOC2012_train_val/VOC2012_train_val")
    parser.add_argument("--splits_json", type=str, default="C:/Users/ARKAJYOTI DAS/Downloads/mini_competition_group_15/archive (11)/voc2012_splits.json")
    parser.add_argument("--checkpoint_path", type=str, default="C:/Users/ARKAJYOTI DAS/Downloads/mini_competition_group_15/best_checkpoint (11).pth")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=300)
    parser.add_argument("--num_classes", type=int, default=21)
    parser.add_argument("--use_dml", action="store_true", help="Use DirectML for Intel GPU on Windows")

    args = parser.parse_args()
    main(args)
