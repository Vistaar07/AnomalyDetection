import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config

class xBDDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.is_train = is_train

        all_files = os.listdir(data_dir)
        self.tile_bases = list(set([
            f.split('_pre.png')[0]
            for f in all_files
            if '_pre.png' in f and 'global' not in f
        ]))

        self.global_cache = {}

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={
            'image2': 'image',
            'global_pre': 'image',
            'global_post': 'image',
            'edge': 'mask'
        })

        # Validation: normalize only, no augmentation.
        self.val_transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={
            'image2': 'image',
            'global_pre': 'image',
            'global_post': 'image',
            'edge': 'mask'
        })

    def __len__(self):
        return len(self.tile_bases)

    def __getitem__(self, idx):
        base_name = self.tile_bases[idx]

        event_name = base_name.split('_')[0] + "_" + base_name.split('_')[1]

        pre  = cv2.imread(os.path.join(self.data_dir, f"{base_name}_pre.png"))[:, :, ::-1]
        post = cv2.imread(os.path.join(self.data_dir, f"{base_name}_post.png"))[:, :, ::-1]
        mask = cv2.imread(os.path.join(self.data_dir, f"{base_name}_mask.png"), cv2.IMREAD_GRAYSCALE)
        edge = cv2.imread(os.path.join(self.data_dir, f"{base_name}_edge.png"), cv2.IMREAD_GRAYSCALE)

        if pre is None or post is None:
            raise ValueError(f"Image load failed: {base_name}")

        # Cache global context images per event to avoid redundant disk reads.
        if event_name not in self.global_cache:
            g_pre  = cv2.imread(os.path.join(self.data_dir, f"{event_name}_global_pre.png"))[:, :, ::-1]
            g_post = cv2.imread(os.path.join(self.data_dir, f"{event_name}_global_post.png"))[:, :, ::-1]
            self.global_cache[event_name] = (g_pre, g_post)

        g_pre, g_post = self.global_cache[event_name]

        tfms = self.transform if self.is_train else self.val_transform

        aug = tfms(
            image=pre,
            image2=post,
            global_pre=g_pre,
            global_post=g_post,
            mask=mask,
            edge=edge
        )

        return {
            'pre':    aug['image'],
            'post':   aug['image2'],
            'g_pre':  aug['global_pre'],
            'g_post': aug['global_post'],
            'mask':   aug['mask'].long(),
            'edge':   aug['edge'].float()
        }