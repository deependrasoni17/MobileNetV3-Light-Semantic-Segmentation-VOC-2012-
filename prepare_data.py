import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
from argparse import ArgumentParser
from torch.utils.data import DataLoader

# Import the necessary components from your dataset.py file
from dataset import VOCFolderSegDataset, get_transforms, list_image_ids, create_and_save_split

def loader_check(dataset, batch_size=4, num_workers=2, n_batches=5):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print("Running loader check...")
    for i, (imgs, masks) in enumerate(loader):
        print(f"Batch {i}: imgs {imgs.shape} dtype {imgs.dtype}; masks {masks.shape} dtype {masks.dtype}")
        mmin = int(masks.min().item())
        mmax = int(masks.max().item())
        print(f"  mask value range: {mmin} .. {mmax}")
        assert imgs.shape[2] == imgs.shape[3] == 300, "Image spatial size must be 300x300"
        assert masks.shape[1] == 300 and masks.shape[2] == 300, "Mask spatial size must be 300x300"
        
        # check label range: allow 0..20 and 255
        if not ((mmin >= 0 and mmax <= 20) or (mmax == 255) or (mmin == 255)):
            raise ValueError("Mask values outside expected range [0,20] or 255 (ignore).")
        
        if i + 1 >= n_batches:
            break
            
    print("Loader check passed.")

def main(args):
    # Delete the existing split file to force regeneration if the flag is passed
    if args.force_resplit and os.path.exists(args.out_splits):
        os.remove(args.out_splits)
        print(f"Removed existing split file: {args.out_splits}")

    voc_root = args.voc_root
    assert os.path.isdir(voc_root), f"VOC root not found: {voc_root}"
    
    jpeg_dir = os.path.join(voc_root, "JPEGImages")
    seg_dir = os.path.join(voc_root, "SegmentationClass")
    assert os.path.isdir(jpeg_dir), f"JPEGImages not found in {voc_root}"
    assert os.path.isdir(seg_dir), f"SegmentationClass not found in {voc_root}"

    all_image_ids = list_image_ids(jpeg_dir)
    print(f"Found {len(all_image_ids)} potential images in {jpeg_dir}")

    # Filter image_ids to ensure corresponding masks exist
    filtered_image_ids = []
    for img_id in all_image_ids:
        # Ensure mask exists for the image_id
        mask_path = os.path.join(seg_dir, img_id + ".png")
        if os.path.exists(mask_path):
            filtered_image_ids.append(img_id)

    image_ids = sorted(list(set(filtered_image_ids))) # Ensure uniqueness and sort
    print(f"Found {len(image_ids)} images with corresponding segmentation masks.")

    # create or load splits
    if os.path.exists(args.out_splits):
        print(f"Loading existing splits from {args.out_splits}")
        with open(args.out_splits, "r") as f:
            splits = json.load(f)
        train_ids = splits["train_ids"]
        val_ids = splits["val_ids"]
    else:
        train_ids, val_ids = create_and_save_split(image_ids, args.train_frac, args.seed, args.out_splits)

    # transforms (unpacking both train and val transforms from dataset.py)
    train_transform, val_transform = get_transforms(
        img_size=args.img_size, corruption_prob=args.corruption_prob
    )

    # datasets
    train_dataset = VOCFolderSegDataset(
        voc_root=voc_root,
        image_ids=train_ids,
        transform=train_transform
    )

    val_dataset = VOCFolderSegDataset(
        voc_root=voc_root,
        image_ids=val_ids,
        transform=val_transform
    )
    
    print(f"Train size: {len(train_dataset)}; Val size: {len(val_dataset)}")

    # loader check
    loader_check(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, n_batches=3)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--voc_root", type=str,
                        default="C:/Users/ARKAJYOTI DAS/Downloads/mini_competition_group_15/archive (11)/VOC2012_train_val/VOC2012_train_val",
                        help="Path to VOC root that contains JPEGImages and SegmentationClass")
    parser.add_argument("--out_splits", type=str,
                        default="C:/Users/ARKAJYOTI DAS/Downloads/mini_competition_group_15/archive (11)/voc2012_splits.json",
                        help="Path to save train/val split JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--img_size", type=int, default=300)
    parser.add_argument("--corruption_prob", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--force_resplit", action="store_true", 
                        help="Include this flag to delete and regenerate the splits file")
    
    # Properly parse arguments for terminal execution
    args = parser.parse_args()
    
    main(args)