"""
Analysis and Visualization Tools for Severstal Steel Defect Detection.
Includes EDA, training curves analysis, and prediction visualization.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

from src.utils.helpers import rle_decode


# ==================== EXPLORATORY DATA ANALYSIS ====================

class DatasetAnalyzer:
    """
    Comprehensive dataset analysis for Severstal Steel Defect Detection.
    """
    
    def __init__(self, csv_path: str, image_dir: str):
        """
        Args:
            csv_path: Path to train.csv
            image_dir: Path to training images
        """
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_path)
        
        # Handle both CSV formats:
        # Format 1: ImageId_ClassId, EncodedPixels (combined)
        # Format 2: ImageId, ClassId, EncodedPixels (separate columns)
        if 'ImageId_ClassId' in self.df.columns:
            self.df['ImageId'] = self.df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
            self.df['ClassId'] = self.df['ImageId_ClassId'].apply(lambda x: int(x.split('_')[1]))
        elif 'ImageId' in self.df.columns and 'ClassId' in self.df.columns:
            self.df['ClassId'] = self.df['ClassId'].astype(int)
        else:
            raise ValueError("CSV must have either 'ImageId_ClassId' or 'ImageId' and 'ClassId' columns")
        self.df['HasDefect'] = self.df['EncodedPixels'].notna().astype(int)
    
    def basic_statistics(self) -> Dict:
        """Get basic dataset statistics."""
        unique_images = self.df['ImageId'].nunique()
        total_rows = len(self.df)
        
        # Images with defects
        images_with_defects = self.df.groupby('ImageId')['HasDefect'].max().sum()
        
        # Per-class statistics
        class_stats = {}
        for class_id in range(1, 5):
            class_df = self.df[self.df['ClassId'] == class_id]
            n_defects = class_df['HasDefect'].sum()
            class_stats[f'Class_{class_id}'] = {
                'count': n_defects,
                'percentage': n_defects / unique_images * 100
            }
        
        # Defect combinations
        defect_combos = self.df[self.df['HasDefect'] == 1].groupby('ImageId')['ClassId'].apply(
            lambda x: '_'.join(map(str, sorted(x)))
        ).value_counts()
        
        stats = {
            'total_images': unique_images,
            'total_rows': total_rows,
            'images_with_defects': images_with_defects,
            'images_without_defects': unique_images - images_with_defects,
            'defect_ratio': images_with_defects / unique_images,
            'class_statistics': class_stats,
            'defect_combinations': defect_combos.to_dict()
        }
        
        return stats
    
    def plot_class_distribution(self, save_path: Optional[str] = None):
        """Plot defect class distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Class counts
        class_counts = self.df.groupby('ClassId')['HasDefect'].sum()
        ax1 = axes[0]
        bars = ax1.bar(class_counts.index, class_counts.values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        ax1.set_xlabel('Defect Class', fontsize=12)
        ax1.set_ylabel('Number of Images', fontsize=12)
        ax1.set_title('Defect Count per Class', fontsize=14)
        ax1.set_xticks([1, 2, 3, 4])
        
        for bar, count in zip(bars, class_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{count}', ha='center', fontsize=10)
        
        # Defect vs No Defect
        ax2 = axes[1]
        images_per_image = self.df.groupby('ImageId')['HasDefect'].max()
        defect_counts = [
            (images_per_image == 0).sum(),
            (images_per_image == 1).sum()
        ]
        colors = ['#95a5a6', '#e74c3c']
        wedges, texts, autotexts = ax2.pie(
            defect_counts, 
            labels=['No Defect', 'Has Defect'],
            autopct='%1.1f%%',
            colors=colors,
            explode=(0, 0.05)
        )
        ax2.set_title('Images with/without Defects', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def plot_defect_area_distribution(self, save_path: Optional[str] = None):
        """Plot distribution of defect areas."""
        defect_areas = []
        
        for _, row in self.df[self.df['HasDefect'] == 1].iterrows():
            mask = rle_decode(row['EncodedPixels'])
            area = mask.sum()
            defect_areas.append({
                'ClassId': row['ClassId'],
                'Area': area,
                'AreaPercentage': area / (256 * 1600) * 100
            })
        
        area_df = pd.DataFrame(defect_areas)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Box plot of areas per class
        ax1 = axes[0]
        area_df.boxplot(column='Area', by='ClassId', ax=ax1)
        ax1.set_xlabel('Defect Class', fontsize=12)
        ax1.set_ylabel('Defect Area (pixels)', fontsize=12)
        ax1.set_title('Defect Area Distribution per Class', fontsize=14)
        plt.suptitle('')
        
        # Histogram of area percentages
        ax2 = axes[1]
        for class_id in range(1, 5):
            class_data = area_df[area_df['ClassId'] == class_id]['AreaPercentage']
            ax2.hist(class_data, bins=50, alpha=0.5, label=f'Class {class_id}')
        ax2.set_xlabel('Defect Area (%)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Defect Area Distribution', fontsize=14)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_samples(
        self,
        n_samples: int = 4,
        class_id: Optional[int] = None,
        save_path: Optional[str] = None
    ):
        """Visualize sample images with defect masks."""
        # Filter by class if specified
        if class_id:
            sample_df = self.df[(self.df['ClassId'] == class_id) & (self.df['HasDefect'] == 1)]
        else:
            sample_df = self.df[self.df['HasDefect'] == 1]
        
        # Sample random images
        sample_images = sample_df['ImageId'].drop_duplicates().sample(n=min(n_samples, len(sample_df)))
        
        fig, axes = plt.subplots(n_samples, 2, figsize=(16, 4 * n_samples))
        if n_samples == 1:
            axes = [axes]
        
        for idx, img_id in enumerate(sample_images):
            # Load image
            img_path = os.path.join(self.image_dir, img_id)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get masks for all classes
            img_df = self.df[self.df['ImageId'] == img_id]
            combined_mask = np.zeros((256, 1600, 3), dtype=np.uint8)
            
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # R, G, B, Y
            defect_classes = []
            
            for _, row in img_df.iterrows():
                if pd.notna(row['EncodedPixels']):
                    mask = rle_decode(row['EncodedPixels'])
                    class_idx = row['ClassId'] - 1
                    combined_mask[mask == 1] = colors[class_idx]
                    defect_classes.append(row['ClassId'])
            
            # Original image
            axes[idx][0].imshow(image)
            axes[idx][0].set_title(f'{img_id}', fontsize=12)
            axes[idx][0].axis('off')
            
            # Image with overlay
            overlay = image.copy()
            alpha = 0.5
            mask_indices = combined_mask.sum(axis=2) > 0
            overlay[mask_indices] = (
                overlay[mask_indices] * (1 - alpha) + 
                combined_mask[mask_indices] * alpha
            ).astype(np.uint8)
            
            axes[idx][1].imshow(overlay)
            axes[idx][1].set_title(f'Defects: {defect_classes}', fontsize=12)
            axes[idx][1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_summary(self):
        """Print comprehensive summary."""
        stats = self.basic_statistics()
        
        print("=" * 60)
        print("SEVERSTAL STEEL DEFECT DETECTION - DATASET ANALYSIS")
        print("=" * 60)
        
        print(f"\n📊 Basic Statistics:")
        print(f"   Total images: {stats['total_images']:,}")
        print(f"   Images with defects: {stats['images_with_defects']:,} ({stats['defect_ratio']*100:.1f}%)")
        print(f"   Images without defects: {stats['images_without_defects']:,}")
        
        print(f"\n📈 Per-Class Distribution:")
        for class_name, class_stat in stats['class_statistics'].items():
            print(f"   {class_name}: {class_stat['count']:,} images ({class_stat['percentage']:.1f}%)")
        
        print(f"\n🔗 Most Common Defect Combinations:")
        for combo, count in list(stats['defect_combinations'].items())[:5]:
            if combo:
                classes = [f"Class {c}" for c in combo.split('_')]
                print(f"   {' + '.join(classes)}: {count} images")


# ==================== TRAINING ANALYSIS ====================

class TrainingAnalyzer:
    """
    Analyze training history and learning curves.
    """
    
    def __init__(self, history_path: str):
        """
        Args:
            history_path: Path to training_history.json
        """
        with open(history_path, 'r') as f:
            self.history = json.load(f)
    
    def plot_learning_curves(self, save_path: Optional[str] = None):
        """Plot training and validation loss curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss curves
        ax1 = axes[0]
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Find best epoch
        best_epoch = np.argmin(self.history['val_loss']) + 1
        best_val_loss = min(self.history['val_loss'])
        ax1.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best: Epoch {best_epoch}')
        ax1.scatter([best_epoch], [best_val_loss], c='g', s=100, zorder=5)
        
        # Dice curves (if available)
        ax2 = axes[1]
        if 'train_metrics' in self.history and len(self.history['train_metrics']) > 0:
            train_dice = [m.get('dice', 0) for m in self.history['train_metrics']]
            val_dice = [m.get('dice', 0) for m in self.history['val_metrics']]
            
            ax2.plot(epochs, train_dice, 'b-', label='Training', linewidth=2)
            ax2.plot(epochs, val_dice, 'r-', label='Validation', linewidth=2)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Dice Score', fontsize=12)
            ax2.set_title('Training and Validation Dice', fontsize=14)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            best_dice_epoch = np.argmax(val_dice) + 1
            best_dice = max(val_dice)
            ax2.axvline(x=best_dice_epoch, color='g', linestyle='--')
            ax2.scatter([best_dice_epoch], [best_dice], c='g', s=100, zorder=5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_learning_rate(self, save_path: Optional[str] = None):
        """Plot learning rate schedule."""
        if 'lr' not in self.history:
            print("No learning rate history found")
            return
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        epochs = range(1, len(self.history['lr']) + 1)
        ax.plot(epochs, self.history['lr'], 'b-', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_summary(self):
        """Print training summary."""
        print("=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        
        n_epochs = len(self.history['train_loss'])
        best_epoch = np.argmin(self.history['val_loss']) + 1
        best_val_loss = min(self.history['val_loss'])
        final_train_loss = self.history['train_loss'][-1]
        final_val_loss = self.history['val_loss'][-1]
        
        print(f"\n📈 Training Progress:")
        print(f"   Total epochs: {n_epochs}")
        print(f"   Best epoch: {best_epoch}")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        print(f"   Final training loss: {final_train_loss:.4f}")
        print(f"   Final validation loss: {final_val_loss:.4f}")
        
        if 'val_metrics' in self.history and len(self.history['val_metrics']) > 0:
            best_dice = max([m.get('dice', 0) for m in self.history['val_metrics']])
            final_dice = self.history['val_metrics'][-1].get('dice', 0)
            print(f"   Best validation Dice: {best_dice:.4f}")
            print(f"   Final validation Dice: {final_dice:.4f}")


# ==================== PREDICTION VISUALIZATION ====================

class PredictionVisualizer:
    """
    Visualize model predictions.
    """
    
    def __init__(self, image_dir: str):
        self.image_dir = image_dir
        self.class_colors = [
            (255, 0, 0),    # Class 1: Red
            (0, 255, 0),    # Class 2: Green
            (0, 0, 255),    # Class 3: Blue
            (255, 255, 0)   # Class 4: Yellow
        ]
    
    def visualize_prediction(
        self,
        image_id: str,
        gt_masks: Optional[np.ndarray] = None,
        pred_masks: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize ground truth and prediction side by side.
        
        Args:
            image_id: Image filename
            gt_masks: Ground truth masks (4, H, W)
            pred_masks: Predicted masks (4, H, W)
            save_path: Optional save path
        """
        # Load image
        img_path = os.path.join(self.image_dir, image_id)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        n_cols = 1 + (gt_masks is not None) + (pred_masks is not None)
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4))
        
        if n_cols == 1:
            axes = [axes]
        
        col_idx = 0
        
        # Original image
        axes[col_idx].imshow(image)
        axes[col_idx].set_title('Original Image', fontsize=12)
        axes[col_idx].axis('off')
        col_idx += 1
        
        # Ground truth
        if gt_masks is not None:
            gt_overlay = self._create_overlay(image, gt_masks)
            axes[col_idx].imshow(gt_overlay)
            axes[col_idx].set_title('Ground Truth', fontsize=12)
            axes[col_idx].axis('off')
            col_idx += 1
        
        # Prediction
        if pred_masks is not None:
            pred_overlay = self._create_overlay(image, pred_masks)
            axes[col_idx].imshow(pred_overlay)
            axes[col_idx].set_title('Prediction', fontsize=12)
            axes[col_idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _create_overlay(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Create image overlay with colored masks."""
        overlay = image.copy()
        
        for class_idx in range(masks.shape[0]):
            mask = masks[class_idx]
            if mask.max() > 0:
                # Resize mask if needed
                if mask.shape != image.shape[:2]:
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                
                color = np.array(self.class_colors[class_idx])
                mask_3d = np.stack([mask] * 3, axis=-1)
                
                overlay = np.where(
                    mask_3d > 0.5,
                    (1 - alpha) * overlay + alpha * color,
                    overlay
                ).astype(np.uint8)
        
        return overlay
    
    def compare_predictions(
        self,
        image_ids: List[str],
        predictions: Dict[str, np.ndarray],
        ground_truths: Optional[Dict[str, np.ndarray]] = None,
        save_dir: Optional[str] = None
    ):
        """Compare predictions for multiple images."""
        for img_id in image_ids:
            pred = predictions.get(img_id)
            gt = ground_truths.get(img_id) if ground_truths else None
            
            save_path = None
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'pred_{img_id}.png')
            
            self.visualize_prediction(img_id, gt, pred, save_path)


# ==================== RESULTS REPORT ====================

def generate_analysis_report(
    csv_path: str,
    image_dir: str,
    history_path: Optional[str] = None,
    output_dir: str = 'analysis_output'
):
    """
    Generate comprehensive analysis report.
    
    Args:
        csv_path: Path to train.csv
        image_dir: Path to images
        history_path: Optional path to training history
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("GENERATING ANALYSIS REPORT")
    print("=" * 60)
    
    # Dataset analysis
    print("\n[1/3] Analyzing dataset...")
    dataset_analyzer = DatasetAnalyzer(csv_path, image_dir)
    dataset_analyzer.print_summary()
    
    print("\n[2/3] Generating visualizations...")
    dataset_analyzer.plot_class_distribution(
        save_path=os.path.join(output_dir, 'class_distribution.png')
    )
    dataset_analyzer.visualize_samples(
        n_samples=4,
        save_path=os.path.join(output_dir, 'sample_defects.png')
    )
    
    # Training analysis (if available)
    if history_path and os.path.exists(history_path):
        print("\n[3/3] Analyzing training...")
        training_analyzer = TrainingAnalyzer(history_path)
        training_analyzer.print_summary()
        training_analyzer.plot_learning_curves(
            save_path=os.path.join(output_dir, 'learning_curves.png')
        )
        training_analyzer.plot_learning_rate(
            save_path=os.path.join(output_dir, 'learning_rate.png')
        )
    else:
        print("\n[3/3] Skipping training analysis (no history file)")
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset and Training Analysis')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to train.csv')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to images')
    parser.add_argument('--history_path', type=str, help='Path to training history JSON')
    parser.add_argument('--output_dir', type=str, default='analysis_output', help='Output directory')
    
    args = parser.parse_args()
    
    generate_analysis_report(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        history_path=args.history_path,
        output_dir=args.output_dir
    )
