import os
import torch
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from model import MobileNetV3LightSeg

# ==========================================
# 1. INFERENCE TRANSFORMS
# ==========================================
def get_inference_transforms(img_size=300):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# ==========================================
# 2. MAIN INFERENCE SCRIPT
# ==========================================
def run_inference(args):
    if args.use_dml:
        import torch_directml
        device = torch_directml.device()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    model = MobileNetV3LightSeg(num_classes=21)
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)

    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    transforms = get_inference_transforms(args.img_size)

    image_files = [f for f in os.listdir(args.in_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"Found {len(image_files)} images for inference...")

    with torch.inference_mode():
        for filename in image_files:
            img_path = os.path.join(args.in_dir, filename)
            image = np.array(Image.open(img_path).convert("RGB"))

            input_tensor = transforms(image=image)["image"].unsqueeze(0).to(device)

            # model in eval mode returns (B, 300, 300) integer mask directly
            preds = model(input_tensor).squeeze(0).cpu().numpy()

            # Convert to binary mask: background (0) -> black, foreground (>0) -> white (255)
            binary_mask = (preds > 0).astype(np.uint8) * 255

            save_path = os.path.join(args.out_dir, filename)
            cv2.imwrite(save_path, binary_mask)

    print(f"Inference complete. Masks saved to {args.out_dir}")

# ==========================================
# CLI
# ==========================================
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in_dir", type=str,
                        default="augmented_test_sample/JPEGImages",
                        help="Path to input test images")
    parser.add_argument("--out_dir", type=str,
                        default="group_15_output",
                        help="Path to save binary masks (folder must be named group_15_output)")
    parser.add_argument("--checkpoint_path", type=str,
                        default="best_checkpoint (11).pth",
                        help="Path to model weights")
    parser.add_argument("--img_size", type=int, default=300,
                        help="Strict 300x300 resolution constraint")
    parser.add_argument("--use_dml", action="store_true", help="Use DirectML for Intel GPU on Windows")

    args = parser.parse_args()
    run_inference(args)
