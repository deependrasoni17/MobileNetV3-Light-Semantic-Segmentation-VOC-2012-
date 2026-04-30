import os
import glob
import json
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2

def list_image_ids(jpeg_dir):
    exts = ("*.jpg", "*.jpeg", "*.png")
    ids = []
    for e in exts:
        for p in glob.glob(os.path.join(jpeg_dir, e)):
            ids.append(os.path.splitext(os.path.basename(p))[0])
    ids = sorted(list(set(ids)))
    return ids

def create_and_save_split(image_ids, train_frac, seed, out_path):
    rng = random.Random(seed)
    ids = image_ids.copy()
    rng.shuffle(ids)
    n_train = int(train_frac * len(ids))
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]
    splits = {"train_ids": train_ids, "val_ids": val_ids}
    with open(out_path, "w") as f:
        json.dump(splits, f)
    return train_ids, val_ids

def get_transforms(img_size=300, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225), corruption_prob=0.3):
    train_transform = A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.85, 1.15), translate_percent=(0.03, 0.03), rotate=(-10, 10), p=0.5,
                 interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05, p=0.5),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.ISONoise(color_shift=(0.01,0.05), intensity=(0.1,0.4), p=0.4),
            A.MotionBlur(blur_limit=3, p=0.3),
            A.MedianBlur(blur_limit=3, p=0.3),
            A.CoarseDropout(p=0.3),
        ], p=corruption_prob),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    return train_transform, val_transform

class VOCFolderSegDataset(Dataset):
    def __init__(self, voc_root, image_ids, jpeg_dir="JPEGImages", seg_dir="SegmentationClass", transform=None):
        self.voc_root = voc_root
        self.image_ids = image_ids
        self.jpeg_dir = os.path.join(voc_root, jpeg_dir)
        self.seg_dir = os.path.join(voc_root, seg_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def _read_image(self, img_id):
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(self.jpeg_dir, img_id + ext)
            if os.path.exists(p):
                return Image.open(p).convert("RGB")
        raise FileNotFoundError(f"Image for id {img_id} not found in {self.jpeg_dir}")

    def _read_mask(self, img_id):
        p = os.path.join(self.seg_dir, img_id + ".png")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Mask for id {img_id} not found in {self.seg_dir}")
        return Image.open(p)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img = self._read_image(img_id)
        mask = self._read_mask(img_id)

        img_np = np.array(img)  
        mask_np = np.array(mask, dtype=np.int32)  

        if self.transform is not None:
            augmented = self.transform(image=img_np, mask=mask_np)
            img_t = augmented["image"]
            mask_t = augmented["mask"].long()
        else:
            img_t = TF.to_tensor(img)
            mask_t = torch.from_numpy(mask_np).long()

        return img_t, mask_t