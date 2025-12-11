"""
Configuration for Severstal Steel Defect Detection Project.
Centralized configuration management for all models and training.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration."""
    csv_path: str = "data/train.csv"
    train_image_dir: str = "data/train_images"
    test_image_dir: str = "data/test_images"
    
    # Image settings
    original_size: Tuple[int, int] = (256, 1600)
    input_size: Tuple[int, int] = (256, 512)  # Resized for training
    
    # Split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Number of classes
    num_classes: int = 4
    
    # Data loading
    batch_size: int = 8
    num_workers: int = 4


@dataclass
class ClassifierConfig:
    """Binary classifier configuration."""
    backbone: str = "efficientnet_b4"
    pretrained: bool = True
    
    # Training
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Scheduler
    scheduler_T0: int = 10
    scheduler_Tmult: int = 2
    scheduler_eta_min: float = 1e-7
    
    # Threshold
    threshold: float = 0.5


@dataclass
class SegmentationConfig:
    """Segmentation model configuration."""
    model_name: str = "unet"
    encoder_name: str = "efficientnet-b4"
    encoder_weights: str = "imagenet"
    
    # Architecture
    in_channels: int = 3
    classes: int = 4
    activation: Optional[str] = None
    attention_type: Optional[str] = "scse"  # Spatial and Channel SE
    
    # Training
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Loss
    loss_name: str = "bce_dice"
    
    # Scheduler
    scheduler_T0: int = 10
    scheduler_Tmult: int = 2
    scheduler_eta_min: float = 1e-7
    
    # Threshold
    threshold: float = 0.5


@dataclass
class EnsembleConfig:
    """Ensemble configuration."""
    # Classifier
    classifier_threshold: float = 0.5
    
    # Segmentation ensemble
    segmentation_threshold: float = 0.5
    model_weights: Optional[List[float]] = None  # Uniform if None
    
    # Post-processing
    min_defect_areas: Dict[int, int] = field(default_factory=lambda: {
        0: 600,   # Class 1
        1: 600,   # Class 2  
        2: 1000,  # Class 3
        3: 2000   # Class 4
    })
    
    # TTA
    use_tta: bool = True
    tta_transforms: List[str] = field(default_factory=lambda: ['original', 'hflip', 'vflip'])


@dataclass
class TrainingConfig:
    """General training configuration."""
    # Device
    device: str = "auto"  # 'auto', 'cuda', 'cpu'
    
    # Mixed precision
    use_amp: bool = True
    
    # Early stopping
    early_stopping_patience: int = 10
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 10
    
    # Reproducibility
    seed: int = 42


@dataclass
class ProjectConfig:
    """Complete project configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Project info
    project_name: str = "severstal_steel_defect_detection"
    version: str = "1.0.0"


# ==================== PRESET CONFIGURATIONS ====================

def get_fast_dev_config() -> ProjectConfig:
    """Configuration for fast development/debugging."""
    config = ProjectConfig()
    
    # Reduce epochs
    config.classifier.epochs = 3
    config.segmentation.epochs = 5
    
    # Smaller batch
    config.data.batch_size = 4
    
    # Simpler model
    config.segmentation.encoder_name = "resnet34"
    config.classifier.backbone = "efficientnet_b3"
    
    return config


def get_full_training_config() -> ProjectConfig:
    """Configuration for full training."""
    config = ProjectConfig()
    
    # More epochs
    config.classifier.epochs = 50
    config.segmentation.epochs = 100
    
    # Larger batch if possible
    config.data.batch_size = 16
    
    return config


def get_kaggle_submission_config() -> ProjectConfig:
    """Configuration optimized for Kaggle submission."""
    config = ProjectConfig()
    
    # Best performing settings
    config.segmentation.model_name = "unet"
    config.segmentation.encoder_name = "se_resnext50_32x4d"
    config.segmentation.attention_type = "scse"
    config.segmentation.loss_name = "bce_focal_dice"
    
    # Strong post-processing
    config.ensemble.min_defect_areas = {
        0: 800,
        1: 800,
        2: 1200,
        3: 2500
    }
    
    # Full TTA
    config.ensemble.tta_transforms = ['original', 'hflip', 'vflip', 'hvflip']
    
    return config


# ==================== ENSEMBLE ARCHITECTURE PRESETS ====================

ENSEMBLE_ARCHITECTURES = {
    'baseline': [
        {'model_name': 'unet', 'encoder_name': 'efficientnet-b4'},
    ],
    'light': [
        {'model_name': 'unet', 'encoder_name': 'efficientnet-b4'},
        {'model_name': 'unet', 'encoder_name': 'resnet50'},
    ],
    'standard': [
        {'model_name': 'unet', 'encoder_name': 'efficientnet-b4'},
        {'model_name': 'unetplusplus', 'encoder_name': 'se_resnext50_32x4d'},
        {'model_name': 'fpn', 'encoder_name': 'resnet50'},
    ],
    'heavy': [
        {'model_name': 'unet', 'encoder_name': 'efficientnet-b4'},
        {'model_name': 'unet', 'encoder_name': 'efficientnet-b5'},
        {'model_name': 'unetplusplus', 'encoder_name': 'se_resnext50_32x4d'},
        {'model_name': 'fpn', 'encoder_name': 'efficientnet-b4'},
        {'model_name': 'deeplabv3plus', 'encoder_name': 'resnet101'},
    ]
}


# ==================== UTILITY FUNCTIONS ====================

def save_config(config: ProjectConfig, path: str):
    """Save configuration to JSON."""
    import json
    from dataclasses import asdict
    
    config_dict = asdict(config)
    
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Config saved to: {path}")


def load_config(path: str) -> ProjectConfig:
    """Load configuration from JSON."""
    import json
    
    with open(path, 'r') as f:
        config_dict = json.load(f)
    
    # Reconstruct nested dataclasses
    config = ProjectConfig(
        data=DataConfig(**config_dict.get('data', {})),
        classifier=ClassifierConfig(**config_dict.get('classifier', {})),
        segmentation=SegmentationConfig(**config_dict.get('segmentation', {})),
        ensemble=EnsembleConfig(**config_dict.get('ensemble', {})),
        training=TrainingConfig(**config_dict.get('training', {})),
        project_name=config_dict.get('project_name', 'severstal'),
        version=config_dict.get('version', '1.0.0')
    )
    
    return config


def print_config(config: ProjectConfig):
    """Pretty print configuration."""
    from dataclasses import asdict
    import json
    
    print("=" * 60)
    print("PROJECT CONFIGURATION")
    print("=" * 60)
    print(json.dumps(asdict(config), indent=2))


# Default configuration
DEFAULT_CONFIG = ProjectConfig()
