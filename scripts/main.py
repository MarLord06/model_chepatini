"""
Main entry point for Severstal Steel Defect Detection Project.
Provides CLI interface for training, prediction, and analysis.
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import random

from configs.config import (
    ProjectConfig, 
    get_fast_dev_config, 
    get_full_training_config,
    get_kaggle_submission_config,
    save_config,
    load_config,
    print_config,
    ENSEMBLE_ARCHITECTURES
)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cmd_analyze(args):
    """Run dataset and training analysis."""
    from src.analysis.analyzer import generate_analysis_report
    
    generate_analysis_report(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        history_path=args.history_path,
        output_dir=args.output_dir
    )


def cmd_train_classifier(args):
    """Train binary classifier."""
    from src.training.trainer import train_classifier
    
    config = load_config(args.config) if args.config else ProjectConfig()
    set_seed(config.training.seed)
    
    model_path = train_classifier(
        csv_path=args.csv_path or config.data.csv_path,
        image_dir=args.image_dir or config.data.train_image_dir,
        backbone=config.classifier.backbone,
        batch_size=config.data.batch_size,
        epochs=args.epochs or config.classifier.epochs,
        lr=config.classifier.learning_rate,
        image_size=config.data.input_size,
        device=config.training.device,
        checkpoint_dir=config.training.checkpoint_dir
    )
    
    print(f"\n✅ Classifier trained successfully!")
    print(f"   Model saved to: {model_path}")


def cmd_train_segmentation(args):
    """Train segmentation model."""
    from src.training.trainer import train_segmentation
    
    config = load_config(args.config) if args.config else ProjectConfig()
    set_seed(config.training.seed)
    
    model_path = train_segmentation(
        csv_path=args.csv_path or config.data.csv_path,
        image_dir=args.image_dir or config.data.train_image_dir,
        model_name=args.model_name or config.segmentation.model_name,
        encoder_name=args.encoder_name or config.segmentation.encoder_name,
        batch_size=config.data.batch_size,
        epochs=args.epochs or config.segmentation.epochs,
        lr=config.segmentation.learning_rate,
        loss_name=config.segmentation.loss_name,
        image_size=config.data.input_size,
        device=config.training.device,
        checkpoint_dir=config.training.checkpoint_dir
    )
    
    print(f"\n✅ Segmentation model trained successfully!")
    print(f"   Model saved to: {model_path}")


def cmd_train_ensemble(args):
    """Train complete ensemble."""
    from src.training.trainer import train_ensemble
    
    config = load_config(args.config) if args.config else ProjectConfig()
    set_seed(config.training.seed)
    
    model_paths = train_ensemble(
        csv_path=args.csv_path or config.data.csv_path,
        image_dir=args.image_dir or config.data.train_image_dir,
        checkpoint_dir=config.training.checkpoint_dir,
        device=config.training.device
    )
    
    print(f"\n✅ Ensemble trained successfully!")
    print(f"   Model paths:")
    for name, path in model_paths.items():
        print(f"     - {name}: {path}")


def cmd_predict(args):
    """Generate predictions/submission."""
    from src.inference.predictor import create_submission
    
    config = load_config(args.config) if args.config else ProjectConfig()
    
    submission_df = create_submission(
        classifier_path=args.classifier,
        segmentation_paths=args.segmentation,
        test_image_dir=args.test_dir or config.data.test_image_dir,
        submission_csv_path=args.output,
        device=config.training.device,
        batch_size=config.data.batch_size,
        use_tta=config.ensemble.use_tta
    )
    
    print(f"\n✅ Submission generated!")
    print(f"   Output: {args.output}")


def cmd_config(args):
    """Manage configuration."""
    if args.action == 'show':
        config = load_config(args.path) if args.path else ProjectConfig()
        print_config(config)
    
    elif args.action == 'create':
        if args.preset == 'fast':
            config = get_fast_dev_config()
        elif args.preset == 'full':
            config = get_full_training_config()
        elif args.preset == 'kaggle':
            config = get_kaggle_submission_config()
        else:
            config = ProjectConfig()
        
        output_path = args.output or 'config.json'
        save_config(config, output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Severstal Steel Defect Detection - Ensemble Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze dataset
  python main.py analyze --csv_path data/train.csv --image_dir data/train_images
  
  # Train classifier
  python main.py train-classifier --csv_path data/train.csv --image_dir data/train_images
  
  # Train segmentation model
  python main.py train-segmentation --csv_path data/train.csv --image_dir data/train_images
  
  # Train complete ensemble
  python main.py train-ensemble --csv_path data/train.csv --image_dir data/train_images
  
  # Generate submission
  python main.py predict --classifier checkpoints/classifier/best.pth \\
                         --segmentation checkpoints/seg1/best.pth checkpoints/seg2/best.pth \\
                         --test_dir data/test_images \\
                         --output submission.csv
  
  # Create configuration file
  python main.py config create --preset kaggle --output my_config.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # ==================== ANALYZE ====================
    analyze_parser = subparsers.add_parser('analyze', help='Analyze dataset and training')
    analyze_parser.add_argument('--csv_path', type=str, required=True, help='Path to train.csv')
    analyze_parser.add_argument('--image_dir', type=str, required=True, help='Path to images')
    analyze_parser.add_argument('--history_path', type=str, help='Path to training history JSON')
    analyze_parser.add_argument('--output_dir', type=str, default='analysis_output', help='Output directory')
    
    # ==================== TRAIN CLASSIFIER ====================
    train_cls_parser = subparsers.add_parser('train-classifier', help='Train binary classifier')
    train_cls_parser.add_argument('--csv_path', type=str, help='Path to train.csv')
    train_cls_parser.add_argument('--image_dir', type=str, help='Path to images')
    train_cls_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_cls_parser.add_argument('--config', type=str, help='Path to config JSON')
    
    # ==================== TRAIN SEGMENTATION ====================
    train_seg_parser = subparsers.add_parser('train-segmentation', help='Train segmentation model')
    train_seg_parser.add_argument('--csv_path', type=str, help='Path to train.csv')
    train_seg_parser.add_argument('--image_dir', type=str, help='Path to images')
    train_seg_parser.add_argument('--model_name', type=str, help='Model name (unet, unetplusplus, fpn, deeplabv3plus)')
    train_seg_parser.add_argument('--encoder_name', type=str, help='Encoder backbone name')
    train_seg_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_seg_parser.add_argument('--config', type=str, help='Path to config JSON')
    
    # ==================== TRAIN ENSEMBLE ====================
    train_ens_parser = subparsers.add_parser('train-ensemble', help='Train complete ensemble')
    train_ens_parser.add_argument('--csv_path', type=str, help='Path to train.csv')
    train_ens_parser.add_argument('--image_dir', type=str, help='Path to images')
    train_ens_parser.add_argument('--config', type=str, help='Path to config JSON')
    
    # ==================== PREDICT ====================
    predict_parser = subparsers.add_parser('predict', help='Generate predictions')
    predict_parser.add_argument('--classifier', type=str, required=True, help='Path to classifier model')
    predict_parser.add_argument('--segmentation', type=str, nargs='+', required=True, help='Paths to segmentation models')
    predict_parser.add_argument('--test_dir', type=str, help='Path to test images')
    predict_parser.add_argument('--output', type=str, default='submission.csv', help='Output CSV path')
    predict_parser.add_argument('--config', type=str, help='Path to config JSON')
    
    # ==================== CONFIG ====================
    config_parser = subparsers.add_parser('config', help='Manage configuration')
    config_parser.add_argument('action', choices=['show', 'create'], help='Config action')
    config_parser.add_argument('--path', type=str, help='Config file path')
    config_parser.add_argument('--preset', type=str, choices=['default', 'fast', 'full', 'kaggle'], default='default', help='Configuration preset')
    config_parser.add_argument('--output', type=str, help='Output path for new config')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # Execute command
    commands = {
        'analyze': cmd_analyze,
        'train-classifier': cmd_train_classifier,
        'train-segmentation': cmd_train_segmentation,
        'train-ensemble': cmd_train_ensemble,
        'predict': cmd_predict,
        'config': cmd_config
    }
    
    print("=" * 60)
    print("SEVERSTAL STEEL DEFECT DETECTION")
    print(f"Command: {args.command}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    commands[args.command](args)


if __name__ == '__main__':
    main()
