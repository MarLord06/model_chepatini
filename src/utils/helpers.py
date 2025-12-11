"""
Utility functions for Severstal Steel Defect Detection
Includes RLE encoding/decoding, image processing, and metrics
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


def rle_decode(mask_rle: str, shape: Tuple[int, int] = (256, 1600)) -> np.ndarray:
    """
    Decode RLE (Run-Length Encoding) string to binary mask.
    
    Args:
        mask_rle: RLE string in format "start1 length1 start2 length2 ..."
        shape: (height, width) of the output mask
    
    Returns:
        Binary mask of shape (height, width)
    """
    if pd.isna(mask_rle) or mask_rle == '':
        return np.zeros(shape, dtype=np.uint8)
    
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(shape, order='F')


def rle_encode(mask: np.ndarray) -> str:
    """
    Encode binary mask to RLE string.
    
    Args:
        mask: Binary mask of shape (height, width)
    
    Returns:
        RLE string or empty string if mask is empty
    """
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    
    if len(runs) == 0:
        return ''
    
    return ' '.join(str(x) for x in runs)


def post_process_mask(mask: np.ndarray, min_size: int = 800, threshold: float = 0.5) -> np.ndarray:
    """
    Post-process predicted mask to remove small false positives.
    
    Args:
        mask: Predicted mask (continuous values or binary)
        min_size: Minimum pixel area to keep a connected component
        threshold: Threshold to binarize continuous masks
    
    Returns:
        Post-processed binary mask
    """
    # Binarize if needed
    if mask.max() <= 1.0 and mask.dtype == np.float32:
        binary_mask = (mask > threshold).astype(np.uint8)
    else:
        binary_mask = mask.astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # Filter by size
    filtered_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered_mask[labels == i] = 1
    
    return filtered_mask


def mask_to_rle(mask: np.ndarray, min_size: int = 800, threshold: float = 0.5) -> str:
    """
    Convert mask to RLE with post-processing.
    Only returns RLE if there's actual content after filtering.
    
    Args:
        mask: Predicted mask
        min_size: Minimum size for connected components
        threshold: Binarization threshold
    
    Returns:
        RLE string or empty string
    """
    processed_mask = post_process_mask(mask, min_size, threshold)
    
    # Only return RLE if there's actual content
    if processed_mask.sum() == 0:
        return ''
    
    return rle_encode(processed_mask)


def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Calculate Dice coefficient between two masks.
    
    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice coefficient (0 to 1)
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    intersection = np.sum(y_true_flat * y_pred_flat)
    union = np.sum(y_true_flat) + np.sum(y_pred_flat)
    
    return (2.0 * intersection + smooth) / (union + smooth)


def calculate_iou(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Calculate Intersection over Union (IoU).
    
    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
        smooth: Smoothing factor
    
    Returns:
        IoU score (0 to 1)
    """
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    
    return (intersection + smooth) / (union + smooth)


def resize_with_aspect_ratio(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image maintaining original aspect ratio information.
    
    Args:
        image: Input image
        target_size: (height, width) target size
    
    Returns:
        Resized image
    """
    return cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)


def apply_tta(image: np.ndarray) -> List[np.ndarray]:
    """
    Apply Test-Time Augmentation transformations.
    
    Args:
        image: Input image
    
    Returns:
        List of augmented images
    """
    augmented = [
        image,  # Original
        np.fliplr(image),  # Horizontal flip
        np.flipud(image),  # Vertical flip
        np.fliplr(np.flipud(image)),  # Both flips
    ]
    return augmented


def reverse_tta(masks: List[np.ndarray]) -> List[np.ndarray]:
    """
    Reverse Test-Time Augmentation transformations on masks.
    
    Args:
        masks: List of predicted masks with TTA applied
    
    Returns:
        List of masks with reversed transformations
    """
    reversed_masks = [
        masks[0],  # Original
        np.fliplr(masks[1]),  # Reverse horizontal flip
        np.flipud(masks[2]),  # Reverse vertical flip
        np.fliplr(np.flipud(masks[3])),  # Reverse both flips
    ]
    return reversed_masks


import pandas as pd

def parse_dataset(csv_path: str, train_dir: str) -> Tuple[pd.DataFrame, dict]:
    """
    Parse Severstal dataset CSV and organize image/mask information.
    
    Args:
        csv_path: Path to train.csv
        train_dir: Directory containing training images
    
    Returns:
        DataFrame with processed data and statistics dictionary
    """
    df = pd.read_csv(csv_path)
    
    # Handle both CSV formats:
    # Format 1: ImageId_ClassId, EncodedPixels (combined)
    # Format 2: ImageId, ClassId, EncodedPixels (separate columns)
    if 'ImageId_ClassId' in df.columns:
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        df['ClassId'] = df['ImageId_ClassId'].apply(lambda x: int(x.split('_')[1]))
    elif 'ImageId' in df.columns and 'ClassId' in df.columns:
        # Already in separate format, ensure ClassId is int
        df['ClassId'] = df['ClassId'].astype(int)
    else:
        raise ValueError("CSV must have either 'ImageId_ClassId' or 'ImageId' and 'ClassId' columns")
    
    # Create binary label: has defect or not
    df['HasDefect'] = df['EncodedPixels'].notna().astype(int)
    
    # Group by image to get per-image statistics
    image_stats = df.groupby('ImageId').agg({
        'HasDefect': 'sum',  # Number of defect classes
        'ClassId': lambda x: list(x)  # List of class IDs
    }).reset_index()
    
    image_stats.rename(columns={'HasDefect': 'NumDefects'}, inplace=True)
    image_stats['HasAnyDefect'] = (image_stats['NumDefects'] > 0).astype(int)
    
    stats = {
        'total_images': len(image_stats),
        'images_with_defects': image_stats['HasAnyDefect'].sum(),
        'images_without_defects': (1 - image_stats['HasAnyDefect']).sum(),
        'defect_distribution': df.groupby('ClassId')['HasDefect'].sum().to_dict()
    }
    
    return df, image_stats, stats
