"""
Advanced Dataset classes for Severstal Steel Defect Detection.
Includes stratified splitting, multi-task datasets, and augmentations.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, List, Optional, Dict
from utils import rle_decode


# ==================== AUGMENTATION CONFIGURATIONS ====================

def get_training_augmentations(image_size: Tuple[int, int] = (256, 512)) -> A.Compose:
    """
    Strong augmentations for training.
    
    Args:
        image_size: (height, width) of output images
    
    Returns:
        Albumentations composition
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT,
            p=0.5
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1),
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.3, p=1),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
            A.ElasticTransform(alpha=1, sigma=50, p=1),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2, p=1),
            A.Sharpen(p=1),
            A.Emboss(p=1),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1
            ),
        ], p=0.3),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3
        ),
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            fill_value=0,
            p=0.2
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])


def get_validation_augmentations(image_size: Tuple[int, int] = (256, 512)) -> A.Compose:
    """
    Minimal augmentations for validation/testing.
    
    Args:
        image_size: (height, width) of output images
    
    Returns:
        Albumentations composition
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])


# ==================== DATASET CLASSES ====================

class SeverstalSegmentationDataset(Dataset):
    """
    Dataset for segmentation task with 4 defect classes.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (256, 1600),
        num_classes: int = 4
    ):
        """
        Args:
            df: DataFrame with ImageId/ClassId columns (or ImageId_ClassId) and EncodedPixels
            image_dir: Directory containing images
            transform: Albumentations transform
            image_size: Original image size for RLE decoding
            num_classes: Number of defect classes
        """
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.image_size = image_size
        self.num_classes = num_classes
        
        # Get unique image IDs
        self.image_ids = df['ImageId'].unique()
        
        # Create lookup for faster access
        self._create_lookup()
    
    def _create_lookup(self):
        """Create efficient lookup for image masks."""
        self.mask_lookup = {}
        for img_id in self.image_ids:
            img_df = self.df[self.df['ImageId'] == img_id]
            masks = {}
            for _, row in img_df.iterrows():
                # Handle both formats
                if 'ClassId' in row.index:
                    class_id = int(row['ClassId']) - 1  # 0-indexed
                else:
                    class_id = int(row['ImageId_ClassId'].split('_')[1]) - 1
                masks[class_id] = row['EncodedPixels']
            self.mask_lookup[img_id] = masks
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_id = self.image_ids[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_id)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create multi-channel mask
        masks = np.zeros((self.image_size[0], self.image_size[1], self.num_classes), dtype=np.float32)
        
        for class_id, rle in self.mask_lookup[img_id].items():
            if pd.notna(rle) and rle != '':
                mask = rle_decode(rle, self.image_size)
                masks[:, :, class_id] = mask
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=masks)
            image = augmented['image']
            masks = augmented['mask']
            
            # Ensure correct dimensions (C, H, W)
            if isinstance(masks, np.ndarray):
                masks = torch.from_numpy(masks.transpose(2, 0, 1)).float()
            elif masks.dim() == 3 and masks.shape[-1] == self.num_classes:
                masks = masks.permute(2, 0, 1).float()
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            masks = torch.from_numpy(masks.transpose(2, 0, 1)).float()
        
        return image, masks


class SeverstalClassificationDataset(Dataset):
    """
    Dataset for binary classification task (has defect or not).
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform: Optional[A.Compose] = None
    ):
        """
        Args:
            df: DataFrame with image info and labels
            image_dir: Directory containing images
            transform: Albumentations transform
        """
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        
        # Get unique images with binary labels
        self.image_data = self._prepare_data()
    
    def _prepare_data(self) -> List[Dict]:
        """Prepare data with binary labels."""
        data = []
        
        image_ids = self.df['ImageId'].unique()
        for img_id in image_ids:
            img_df = self.df[self.df['ImageId'] == img_id]
            has_defect = img_df['EncodedPixels'].notna().any()
            
            # Also track which classes have defects
            defect_classes = []
            for _, row in img_df.iterrows():
                if pd.notna(row['EncodedPixels']):
                    # Handle both formats
                    if 'ClassId' in row.index:
                        class_id = int(row['ClassId'])
                    else:
                        class_id = int(row['ImageId_ClassId'].split('_')[1])
                    defect_classes.append(class_id)
            
            data.append({
                'image_id': img_id,
                'has_defect': int(has_defect),
                'defect_classes': defect_classes
            })
        
        return data
    
    def __len__(self) -> int:
        return len(self.image_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.image_data[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, item['image_id'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        return image, item['has_defect']


class SeverstalMultiTaskDataset(Dataset):
    """
    Dataset that provides both classification labels and segmentation masks.
    Useful for joint training or cascaded models.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (256, 1600),
        num_classes: int = 4
    ):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.image_size = image_size
        self.num_classes = num_classes
        
        self.image_ids = df['ImageId'].unique()
        self._create_lookup()
    
    def _create_lookup(self):
        """Create efficient lookup."""
        self.data_lookup = {}
        
        for img_id in self.image_ids:
            img_df = self.df[self.df['ImageId'] == img_id]
            masks_rle = {}
            has_defect = False
            class_labels = np.zeros(self.num_classes, dtype=np.float32)
            
            for _, row in img_df.iterrows():
                # Handle both formats
                if 'ClassId' in row.index:
                    class_id = int(row['ClassId']) - 1
                else:
                    class_id = int(row['ImageId_ClassId'].split('_')[1]) - 1
                if pd.notna(row['EncodedPixels']) and row['EncodedPixels'] != '':
                    masks_rle[class_id] = row['EncodedPixels']
                    has_defect = True
                    class_labels[class_id] = 1.0
            
            self.data_lookup[img_id] = {
                'masks_rle': masks_rle,
                'has_defect': int(has_defect),
                'class_labels': class_labels
            }
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        img_id = self.image_ids[idx]
        data = self.data_lookup[img_id]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_id)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create masks
        masks = np.zeros((self.image_size[0], self.image_size[1], self.num_classes), dtype=np.float32)
        for class_id, rle in data['masks_rle'].items():
            masks[:, :, class_id] = rle_decode(rle, self.image_size)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=masks)
            image = augmented['image']
            masks = augmented['mask']
            
            if isinstance(masks, np.ndarray):
                masks = torch.from_numpy(masks.transpose(2, 0, 1)).float()
            elif masks.dim() == 3 and masks.shape[-1] == self.num_classes:
                masks = masks.permute(2, 0, 1).float()
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            masks = torch.from_numpy(masks.transpose(2, 0, 1)).float()
        
        return {
            'image': image,
            'masks': masks,
            'has_defect': torch.tensor(data['has_defect'], dtype=torch.float32),
            'class_labels': torch.tensor(data['class_labels'], dtype=torch.float32),
            'image_id': img_id
        }


# ==================== DATA SPLITTING ====================

def create_stratified_split(
    csv_path: str,
    image_dir: Optional[str] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    include_no_defect: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test split based on defect presence.
    
    Args:
        csv_path: Path to train.csv
        image_dir: Optional path to image directory (to include images without defects)
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_state: Random seed
        include_no_defect: If True and image_dir provided, include images without defects
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    df = pd.read_csv(csv_path)
    
    # Handle both CSV formats:
    # Format 1: ImageId_ClassId, EncodedPixels (combined)
    # Format 2: ImageId, ClassId, EncodedPixels (separate columns)
    if 'ImageId_ClassId' in df.columns:
        # Format 1: Extract ImageId from combined column
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        df['ClassId'] = df['ImageId_ClassId'].apply(lambda x: int(x.split('_')[1]))
    elif 'ImageId' not in df.columns:
        raise ValueError("CSV must have either 'ImageId_ClassId' or 'ImageId' column")
    
    # Get images with defects
    images_with_defects = set(df['ImageId'].unique())
    
    # If image_dir is provided, find images without defects
    if image_dir and include_no_defect and os.path.exists(image_dir):
        all_images = set(f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png')))
        images_without_defects = all_images - images_with_defects
        
        # Add rows for images without defects (one row per class with NaN EncodedPixels)
        no_defect_rows = []
        for img_id in images_without_defects:
            for class_id in range(1, 5):  # Classes 1-4
                no_defect_rows.append({
                    'ImageId': img_id,
                    'ClassId': class_id,
                    'EncodedPixels': np.nan
                })
        
        if no_defect_rows:
            no_defect_df = pd.DataFrame(no_defect_rows)
            df = pd.concat([df, no_defect_df], ignore_index=True)
            print(f"Added {len(images_without_defects)} images without defects")
    
    # Create stratification label based on defect pattern
    image_labels = df.groupby('ImageId').apply(
        lambda x: '_'.join(
            sorted([
                str(row['ClassId'])
                for _, row in x.iterrows()
                if pd.notna(row['EncodedPixels'])
            ])
        ) if x['EncodedPixels'].notna().any() else 'no_defect',
        include_groups=False
    ).reset_index()
    image_labels.columns = ['ImageId', 'StratifyLabel']
    
    # Handle rare combinations by grouping them - need enough samples for all splits
    min_samples_needed = max(5, int(3 / min(train_ratio, val_ratio, test_ratio)))
    label_counts = image_labels['StratifyLabel'].value_counts()
    rare_labels = label_counts[label_counts < min_samples_needed].index
    image_labels.loc[image_labels['StratifyLabel'].isin(rare_labels), 'StratifyLabel'] = 'rare_combo'
    
    # Re-check after grouping - if rare_combo still too small, use simpler stratification
    label_counts = image_labels['StratifyLabel'].value_counts()
    min_count = label_counts.min()
    
    if min_count < 2:
        # Fallback to simpler binary stratification (has defect or not)
        print("Warning: Using simplified stratification (has_defect/no_defect)")
        image_labels['StratifyLabel'] = image_labels['StratifyLabel'].apply(
            lambda x: 'no_defect' if x == 'no_defect' else 'has_defect'
        )
    
    # First split: train vs (val + test)
    try:
        train_ids, temp_ids = train_test_split(
            image_labels['ImageId'].values,
            test_size=(val_ratio + test_ratio),
            stratify=image_labels['StratifyLabel'].values,
            random_state=random_state
        )
    except ValueError:
        # Fallback: no stratification
        print("Warning: Stratification failed, using random split")
        train_ids, temp_ids = train_test_split(
            image_labels['ImageId'].values,
            test_size=(val_ratio + test_ratio),
            random_state=random_state
        )
    
    # Second split: val vs test
    temp_labels = image_labels[image_labels['ImageId'].isin(temp_ids)]
    
    # Re-check stratification for second split
    temp_label_counts = temp_labels['StratifyLabel'].value_counts()
    can_stratify = temp_label_counts.min() >= 2 if len(temp_label_counts) > 0 else False
    
    try:
        if can_stratify:
            val_ids, test_ids = train_test_split(
                temp_labels['ImageId'].values,
                test_size=test_ratio / (val_ratio + test_ratio),
                stratify=temp_labels['StratifyLabel'].values,
                random_state=random_state
            )
        else:
            val_ids, test_ids = train_test_split(
                temp_labels['ImageId'].values,
                test_size=test_ratio / (val_ratio + test_ratio),
                random_state=random_state
            )
    except ValueError:
        # Final fallback
        val_ids, test_ids = train_test_split(
            temp_labels['ImageId'].values,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=random_state
        )
    
    # Create final DataFrames
    train_df = df[df['ImageId'].isin(train_ids)].copy()
    val_df = df[df['ImageId'].isin(val_ids)].copy()
    test_df = df[df['ImageId'].isin(test_ids)].copy()
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_ids)} images ({len(train_df)} rows)")
    print(f"  Val: {len(val_ids)} images ({len(val_df)} rows)")
    print(f"  Test: {len(test_ids)} images ({len(test_df)} rows)")
    
    # Print defect distribution
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        n_with_defect = split_df.groupby('ImageId')['EncodedPixels'].apply(
            lambda x: x.notna().any()
        ).sum()
        n_total = len(split_df['ImageId'].unique())
        print(f"  {name} defect ratio: {n_with_defect}/{n_total} = {n_with_defect/n_total:.2%}")
    
    return train_df, val_df, test_df


def create_kfold_splits(
    csv_path: str,
    n_folds: int = 5,
    random_state: int = 42
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create K-fold stratified splits.
    
    Args:
        csv_path: Path to train.csv
        n_folds: Number of folds
        random_state: Random seed
    
    Returns:
        List of (train_df, val_df) tuples for each fold
    """
    df = pd.read_csv(csv_path)
    
    # Handle both CSV formats
    if 'ImageId_ClassId' in df.columns:
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        df['ClassId'] = df['ImageId_ClassId'].apply(lambda x: int(x.split('_')[1]))
    elif 'ImageId' not in df.columns:
        raise ValueError("CSV must have either 'ImageId_ClassId' or 'ImageId' column")
    
    # Create stratification labels
    image_labels = df.groupby('ImageId').apply(
        lambda x: int(x['EncodedPixels'].notna().any())
    ).reset_index()
    image_labels.columns = ['ImageId', 'HasDefect']
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    folds = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(
        image_labels['ImageId'], image_labels['HasDefect']
    )):
        train_ids = image_labels.iloc[train_idx]['ImageId'].values
        val_ids = image_labels.iloc[val_idx]['ImageId'].values
        
        train_df = df[df['ImageId'].isin(train_ids)].copy()
        val_df = df[df['ImageId'].isin(val_ids)].copy()
        
        print(f"Fold {fold + 1}: Train={len(train_ids)}, Val={len(val_ids)}")
        folds.append((train_df, val_df))
    
    return folds


# ==================== DATA LOADERS ====================

def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (256, 512),
    task: str = 'segmentation'
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        image_dir: Image directory
        batch_size: Batch size
        num_workers: Number of workers
        image_size: Image size
        task: 'segmentation', 'classification', or 'multitask'
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_transform = get_training_augmentations(image_size)
    val_transform = get_validation_augmentations(image_size)
    
    if task == 'segmentation':
        train_dataset = SeverstalSegmentationDataset(
            train_df, image_dir, train_transform
        )
        val_dataset = SeverstalSegmentationDataset(
            val_df, image_dir, val_transform
        )
    elif task == 'classification':
        train_dataset = SeverstalClassificationDataset(
            train_df, image_dir, train_transform
        )
        val_dataset = SeverstalClassificationDataset(
            val_df, image_dir, val_transform
        )
    elif task == 'multitask':
        train_dataset = SeverstalMultiTaskDataset(
            train_df, image_dir, train_transform
        )
        val_dataset = SeverstalMultiTaskDataset(
            val_df, image_dir, val_transform
        )
    else:
        raise ValueError(f"Unknown task: {task}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
