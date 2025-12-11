"""
Visualize predictions from the trained models.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def rle_decode(mask_rle: str, shape: tuple = (256, 1600)) -> np.ndarray:
    """
    Decode RLE-encoded mask.
    
    Args:
        mask_rle: RLE string
        shape: (height, width) of the mask
    
    Returns:
        Binary mask (H, W)
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


def visualize_predictions(
    submission_path: str,
    test_image_dir: str,
    num_images: int = 10,
    save_dir: str = None,
    show_only_defects: bool = True
):
    """
    Visualize predictions from submission file.
    
    Args:
        submission_path: Path to submission.csv
        test_image_dir: Directory with test images
        num_images: Number of images to visualize
        save_dir: Directory to save visualizations (None to display)
        show_only_defects: Only show images with detected defects
    """
    # Load submission
    df = pd.read_csv(submission_path)
    print(f"Loaded {len(df)} predictions")
    
    # Parse ImageId and ClassId
    df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
    df['ClassId'] = df['ImageId_ClassId'].apply(lambda x: int(x.split('_')[1]))
    
    # Get unique images
    if show_only_defects:
        # Filter to images with at least one defect
        images_with_defects = df[df['EncodedPixels'].notna()]['ImageId'].unique()
        print(f"Found {len(images_with_defects)} images with defects")
        selected_images = images_with_defects[:num_images]
    else:
        selected_images = df['ImageId'].unique()[:num_images]
    
    # Color map for defect classes
    colors = {
        1: (255, 0, 0),      # Red - Class 1
        2: (0, 255, 0),      # Green - Class 2
        3: (0, 0, 255),      # Blue - Class 3
        4: (255, 255, 0)     # Yellow - Class 4
    }
    
    class_names = {
        1: 'Defect 1',
        2: 'Defect 2',
        3: 'Defect 3',
        4: 'Defect 4'
    }
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    for idx, image_id in enumerate(selected_images):
        print(f"Processing {idx + 1}/{len(selected_images)}: {image_id}")
        
        # Load image
        image_path = os.path.join(test_image_dir, image_id)
        if not os.path.exists(image_path):
            print(f"  Image not found: {image_path}")
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Could not load image: {image_path}")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(20, 8))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title(f'Original: {image_id}', fontsize=14)
        axes[0].axis('off')
        
        # Image with overlay
        overlay = image.copy()
        detected_classes = []
        
        for class_id in range(1, 5):
            # Get prediction for this class
            row = df[(df['ImageId'] == image_id) & (df['ClassId'] == class_id)]
            if len(row) == 0:
                continue
            
            rle = row['EncodedPixels'].values[0]
            if pd.isna(rle) or rle == '':
                continue
            
            # Decode mask
            mask = rle_decode(rle, shape=(h, w))
            
            if mask.sum() > 0:
                detected_classes.append(class_id)
                # Apply colored overlay
                color = colors[class_id]
                mask_bool = mask == 1
                overlay[mask_bool, 0] = (overlay[mask_bool, 0] * 0.5 + color[0] * 0.5).astype(np.uint8)
                overlay[mask_bool, 1] = (overlay[mask_bool, 1] * 0.5 + color[1] * 0.5).astype(np.uint8)
                overlay[mask_bool, 2] = (overlay[mask_bool, 2] * 0.5 + color[2] * 0.5).astype(np.uint8)
                # Draw contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 2)
        
        axes[1].imshow(overlay)
        
        # Build title with detected classes
        if detected_classes:
            classes_str = ', '.join([class_names[c] for c in detected_classes])
            title = f'Predictions: {classes_str}'
        else:
            title = 'No defects detected'
        
        axes[1].set_title(title, fontsize=14)
        axes[1].axis('off')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color=np.array(colors[i])/255, linewidth=4, label=class_names[i])
            for i in range(1, 5)
        ]
        axes[1].legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, f'pred_{image_id.replace(".jpg", ".png")}')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved to: {save_path}")
            plt.close()
        else:
            plt.show()
    
    print(f"\nVisualization complete! Processed {len(selected_images)} images.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', type=str, default='submission.csv',
                        help='Path to submission.csv')
    parser.add_argument('--test_dir', type=str, default='data/test_images',
                        help='Directory with test images')
    parser.add_argument('--num_images', type=int, default=10,
                        help='Number of images to visualize')
    parser.add_argument('--save_dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--show_all', action='store_true',
                        help='Show all images, not just those with defects')
    
    args = parser.parse_args()
    
    visualize_predictions(
        submission_path=args.submission,
        test_image_dir=args.test_dir,
        num_images=args.num_images,
        save_dir=args.save_dir,
        show_only_defects=not args.show_all
    )
