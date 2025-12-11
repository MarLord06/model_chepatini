"""
Comprehensive Training Script for Severstal Steel Defect Detection.
Supports training classifier and segmentation models with advanced features.
"""

import os
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Local imports
from src.data.dataset import (
    create_stratified_split,
    create_dataloaders,
    get_training_augmentations,
    get_validation_augmentations
)
from src.models.classifier import DefectClassifier, MultiHeadClassifier
from src.models.segmentation import get_segmentation_model
from src.models.losses import get_loss


# ==================== METRICS ====================

def calculate_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """Calculate Dice coefficient."""
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    return ((2.0 * intersection + smooth) / (union + smooth)).item()


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """Calculate IoU."""
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    return ((intersection + smooth) / (union + smooth)).item()


def calculate_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate multiple metrics."""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    
    return {
        'dice': calculate_dice(pred_binary, target),
        'iou': calculate_iou(pred_binary, target)
    }


# ==================== TRAINING FUNCTIONS ====================

class Trainer:
    """
    Unified trainer for both classification and segmentation models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        task: str = 'segmentation',
        use_amp: bool = True,
        checkpoint_dir: str = 'checkpoints',
        experiment_name: str = 'experiment'
    ):
        """
        Args:
            model: Model to train
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            task: 'segmentation' or 'classification'
            use_amp: Use automatic mixed precision
            checkpoint_dir: Directory for checkpoints
            experiment_name: Name for this experiment
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.task = task
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # AMP scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'lr': []
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        metrics_sum = {'dice': 0.0, 'iou': 0.0}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch in pbar:
            if self.task == 'segmentation':
                images, masks = batch
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    loss.backward()
                    self.optimizer.step()
                
                # Calculate metrics
                with torch.no_grad():
                    batch_metrics = calculate_metrics(outputs, masks)
                    for k, v in batch_metrics.items():
                        metrics_sum[k] += v
            
            elif self.task == 'classification':
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)
                
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                
                # Calculate accuracy
                with torch.no_grad():
                    pred = torch.sigmoid(outputs) > 0.5
                    correct = (pred == labels).float().mean()
                    metrics_sum['dice'] += correct.item()  # Using dice key for accuracy
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        metrics_sum = {'dice': 0.0, 'iou': 0.0}
        num_batches = 0
        
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        
        for batch in pbar:
            if self.task == 'segmentation':
                images, masks = batch
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                batch_metrics = calculate_metrics(outputs, masks)
                for k, v in batch_metrics.items():
                    metrics_sum[k] += v
            
            elif self.task == 'classification':
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                pred = torch.sigmoid(outputs) > 0.5
                correct = (pred == labels).float().mean()
                metrics_sum['dice'] += correct.item()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping_patience: int = 10
    ) -> Dict[str, List]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
        
        Returns:
            Training history
        """
        no_improvement_count = 0
        
        print(f"\nStarting training: {self.task}")
        print(f"Device: {self.device}")
        print(f"AMP: {self.use_amp}")
        print(f"Checkpoints: {self.checkpoint_dir}")
        print("-" * 50)
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Train
            train_results = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_results = self.validate(val_loader, epoch)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_results['loss'])
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_results['loss'])
            self.history['val_loss'].append(val_results['loss'])
            self.history['train_metrics'].append(train_results)
            self.history['val_metrics'].append(val_results)
            self.history['lr'].append(current_lr)
            
            # Print epoch summary
            elapsed = time.time() - start_time
            print(f"\nEpoch {epoch}/{epochs} ({elapsed:.1f}s)")
            print(f"  Train - Loss: {train_results['loss']:.4f}, Dice: {train_results['dice']:.4f}")
            print(f"  Val   - Loss: {val_results['loss']:.4f}, Dice: {val_results['dice']:.4f}")
            print(f"  LR: {current_lr:.2e}")
            
            # Check for improvement
            if val_results['loss'] < self.best_val_loss:
                self.best_val_loss = val_results['loss']
                self.best_epoch = epoch
                no_improvement_count = 0
                
                # Save best model
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ New best model saved!")
            else:
                no_improvement_count += 1
                print(f"  No improvement for {no_improvement_count} epochs")
            
            # Early stopping
            if no_improvement_count >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print(f"\nTraining complete!")
        print(f"Best epoch: {self.best_epoch}, Best val loss: {self.best_val_loss:.4f}")
        
        # Save training history
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint.get('epoch', 0)
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / 'training_history.json'
        
        # Convert numpy arrays to lists for JSON serialization
        history_json = {}
        for key, value in self.history.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict):
                    history_json[key] = value
                else:
                    history_json[key] = [float(v) for v in value]
            else:
                history_json[key] = value
        
        with open(history_path, 'w') as f:
            json.dump(history_json, f, indent=2)


# ==================== MAIN TRAINING FUNCTIONS ====================

def train_classifier(
    csv_path: str,
    image_dir: str,
    backbone: str = 'efficientnet_b4',
    batch_size: int = 16,
    epochs: int = 50,
    lr: float = 1e-4,
    image_size: Tuple[int, int] = (256, 512),
    device: str = 'auto',
    checkpoint_dir: str = 'checkpoints'
) -> str:
    """
    Train binary defect classifier.
    
    Args:
        csv_path: Path to train.csv
        image_dir: Path to images
        backbone: Backbone architecture
        batch_size: Batch size
        epochs: Number of epochs
        lr: Learning rate
        image_size: Image size
        device: Device ('auto', 'cuda', 'cpu')
        checkpoint_dir: Checkpoint directory
    
    Returns:
        Path to best model
    """
    print("=" * 60)
    print("TRAINING BINARY CLASSIFIER")
    print("=" * 60)
    
    # Setup device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # Create data splits (include images without defects for classifier)
    train_df, val_df, _ = create_stratified_split(
        csv_path, 
        image_dir=image_dir,
        include_no_defect=True
    )
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_df, val_df, image_dir,
        batch_size=batch_size,
        image_size=image_size,
        task='classification'
    )
    
    # Calculate class weights for imbalanced data
    train_labels = train_df.groupby('ImageId')['EncodedPixels'].apply(
        lambda x: int(x.notna().any())
    )
    pos_ratio = train_labels.mean()
    pos_weight = (1 - pos_ratio) / pos_ratio
    print(f"Positive ratio: {pos_ratio:.2%}, pos_weight: {pos_weight:.2f}")
    
    # Create model
    model = DefectClassifier(backbone=backbone, pretrained=True)
    
    # Create loss with class weighting
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(device)
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )
    
    # Create trainer
    experiment_name = f"classifier_{backbone}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        task='classification',
        checkpoint_dir=checkpoint_dir,
        experiment_name=experiment_name
    )
    
    # Train
    trainer.fit(train_loader, val_loader, epochs=epochs)
    
    return str(trainer.checkpoint_dir / 'best_model.pth')


def train_segmentation(
    csv_path: str,
    image_dir: str,
    model_name: str = 'unet',
    encoder_name: str = 'efficientnet-b4',
    batch_size: int = 8,
    epochs: int = 100,
    lr: float = 1e-4,
    loss_name: str = 'bce_dice',
    image_size: Tuple[int, int] = (256, 512),
    device: str = 'auto',
    checkpoint_dir: str = 'checkpoints'
) -> str:
    """
    Train segmentation model.
    
    Args:
        csv_path: Path to train.csv
        image_dir: Path to images
        model_name: Segmentation model type
        encoder_name: Encoder backbone
        batch_size: Batch size
        epochs: Number of epochs
        lr: Learning rate
        loss_name: Loss function name
        image_size: Image size
        device: Device
        checkpoint_dir: Checkpoint directory
    
    Returns:
        Path to best model
    """
    print("=" * 60)
    print("TRAINING SEGMENTATION MODEL")
    print("=" * 60)
    
    # Setup device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # Create data splits (only images with defects for segmentation)
    train_df, val_df, _ = create_stratified_split(
        csv_path,
        image_dir=None,  # Don't include no-defect images for segmentation
        include_no_defect=False
    )
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_df, val_df, image_dir,
        batch_size=batch_size,
        image_size=image_size,
        task='segmentation'
    )
    
    # Create model
    model = get_segmentation_model(
        model_name=model_name,
        encoder_name=encoder_name,
        encoder_weights='imagenet',
        in_channels=3,
        classes=4,
        activation=None
    )
    
    # Create loss
    criterion = get_loss(loss_name)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )
    
    # Create trainer
    experiment_name = f"segmentation_{model_name}_{encoder_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        task='segmentation',
        checkpoint_dir=checkpoint_dir,
        experiment_name=experiment_name
    )
    
    # Train
    trainer.fit(train_loader, val_loader, epochs=epochs)
    
    return str(trainer.checkpoint_dir / 'best_model.pth')


def train_ensemble(
    csv_path: str,
    image_dir: str,
    checkpoint_dir: str = 'checkpoints',
    device: str = 'auto'
) -> Dict[str, str]:
    """
    Train complete ensemble: classifier + multiple segmentation models.
    
    Returns:
        Dictionary of model paths
    """
    print("=" * 60)
    print("TRAINING COMPLETE ENSEMBLE")
    print("=" * 60)
    
    model_paths = {}
    
    # 1. Train binary classifier
    print("\n[1/4] Training Binary Classifier...")
    model_paths['classifier'] = train_classifier(
        csv_path=csv_path,
        image_dir=image_dir,
        backbone='efficientnet_b4',
        epochs=30,
        checkpoint_dir=checkpoint_dir
    )
    
    # 2. Train U-Net with EfficientNet-B4
    print("\n[2/4] Training U-Net with EfficientNet-B4...")
    model_paths['unet_effb4'] = train_segmentation(
        csv_path=csv_path,
        image_dir=image_dir,
        model_name='unet',
        encoder_name='efficientnet-b4',
        epochs=50,
        loss_name='bce_dice',
        checkpoint_dir=checkpoint_dir
    )
    
    # 3. Train U-Net++ with SE-ResNeXt50
    print("\n[3/4] Training U-Net++ with SE-ResNeXt50...")
    model_paths['unetpp_seresnext'] = train_segmentation(
        csv_path=csv_path,
        image_dir=image_dir,
        model_name='unetplusplus',
        encoder_name='se_resnext50_32x4d',
        epochs=50,
        loss_name='bce_dice',
        checkpoint_dir=checkpoint_dir
    )
    
    # 4. Train FPN with ResNet50
    print("\n[4/4] Training FPN with ResNet50...")
    model_paths['fpn_resnet50'] = train_segmentation(
        csv_path=csv_path,
        image_dir=image_dir,
        model_name='fpn',
        encoder_name='resnet50',
        epochs=50,
        loss_name='focal_tversky',
        checkpoint_dir=checkpoint_dir
    )
    
    # Save model paths
    paths_file = Path(checkpoint_dir) / 'ensemble_model_paths.json'
    with open(paths_file, 'w') as f:
        json.dump(model_paths, f, indent=2)
    
    print("\n" + "=" * 60)
    print("ENSEMBLE TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model paths saved to: {paths_file}")
    
    return model_paths


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(description='Train Severstal Steel Defect Detection Models')
    
    parser.add_argument('--mode', type=str, default='ensemble',
                        choices=['classifier', 'segmentation', 'ensemble'],
                        help='Training mode')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to train.csv')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Path to training images')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cuda, cpu)')
    parser.add_argument('--model_name', type=str, default='unet',
                        help='Segmentation model name')
    parser.add_argument('--encoder_name', type=str, default='efficientnet-b4',
                        help='Encoder backbone name')
    
    args = parser.parse_args()
    
    if args.mode == 'classifier':
        train_classifier(
            csv_path=args.csv_path,
            image_dir=args.image_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir
        )
    
    elif args.mode == 'segmentation':
        train_segmentation(
            csv_path=args.csv_path,
            image_dir=args.image_dir,
            model_name=args.model_name,
            encoder_name=args.encoder_name,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir
        )
    
    elif args.mode == 'ensemble':
        train_ensemble(
            csv_path=args.csv_path,
            image_dir=args.image_dir,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device
        )


if __name__ == '__main__':
    main()
