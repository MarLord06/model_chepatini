import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

class SeverstalDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment

        self.transform = A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.485,0.456,0.406),
                        std=(0.229,0.224,0.225)),
        ])

        self.no_aug = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.485,0.456,0.406),
                        std=(0.229,0.224,0.225)),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])[:, :, ::-1]

        if self.mask_paths is not None:
            mask = cv2.imread(self.mask_paths[idx], 0)
            mask = cv2.resize(mask, (512, 512))
            mask = mask / 255.0
            mask = np.expand_dims(mask, 2)

        if self.augment and self.mask_paths is not None:
            data = self.transform(image=img, mask=mask)
        else:
            data = self.no_aug(image=img, mask=mask if self.mask_paths else None)

        img = data["image"].transpose(2, 0, 1)

        if self.mask_paths:
            mask = data["mask"].transpose(2, 0, 1)
            return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

        return torch.tensor(img, dtype=torch.float32)
