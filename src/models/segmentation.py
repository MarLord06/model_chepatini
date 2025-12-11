"""
Advanced Segmentation Models for Steel Defect Detection.
Includes U-Net with various backbones and attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Optional


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AttentionBlock(nn.Module):
    """Attention gate for U-Net decoder."""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionBlock, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNetSegmentation(nn.Module):
    """
    U-Net with modern encoder backbones using segmentation_models_pytorch.
    Supports multiple architectures and attention mechanisms.
    """
    
    def __init__(
        self,
        encoder_name: str = 'efficientnet-b4',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 4,
        activation: Optional[str] = None,
        attention_type: Optional[str] = None
    ):
        """
        Args:
            encoder_name: Encoder backbone ('efficientnet-b4', 'resnet50', 'se_resnext50_32x4d', etc.)
            encoder_weights: Pretrained weights ('imagenet' or None)
            in_channels: Number of input channels
            classes: Number of output classes
            activation: Output activation ('sigmoid', 'softmax', None)
            attention_type: Attention mechanism ('scse' or None)
        """
        super(UNetSegmentation, self).__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            decoder_attention_type=attention_type
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UNetPlusPlus(nn.Module):
    """
    U-Net++ (Nested U-Net) for improved segmentation.
    """
    
    def __init__(
        self,
        encoder_name: str = 'efficientnet-b4',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 4,
        activation: Optional[str] = None
    ):
        super(UNetPlusPlus, self).__init__()
        
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FPN(nn.Module):
    """
    Feature Pyramid Network for multi-scale segmentation.
    """
    
    def __init__(
        self,
        encoder_name: str = 'efficientnet-b4',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 4,
        activation: Optional[str] = None
    ):
        super(FPN, self).__init__()
        
        self.model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ with ASPP for capturing multi-scale context.
    """
    
    def __init__(
        self,
        encoder_name: str = 'efficientnet-b4',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 4,
        activation: Optional[str] = None
    ):
        super(DeepLabV3Plus, self).__init__()
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiScaleUNet(nn.Module):
    """
    Multi-scale U-Net that combines predictions at different resolutions.
    """
    
    def __init__(
        self,
        encoder_name: str = 'efficientnet-b4',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 4
    ):
        super(MultiScaleUNet, self).__init__()
        
        # Main U-Net
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None
        )
        
        # Auxiliary outputs at different scales
        encoder_channels = self.unet.encoder.out_channels
        
        # Deep supervision heads
        self.aux_head1 = nn.Conv2d(encoder_channels[-2], classes, kernel_size=1)
        self.aux_head2 = nn.Conv2d(encoder_channels[-3], classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get encoder features
        features = self.unet.encoder(x)
        
        # Main decoder output
        decoder_output = self.unet.decoder(*features)
        main_output = self.unet.segmentation_head(decoder_output)
        
        return main_output


def get_segmentation_model(
    model_name: str = 'unet',
    encoder_name: str = 'efficientnet-b4',
    encoder_weights: str = 'imagenet',
    in_channels: int = 3,
    classes: int = 4,
    activation: Optional[str] = None,
    attention_type: Optional[str] = None
) -> nn.Module:
    """
    Factory function to create segmentation models.
    
    Args:
        model_name: 'unet', 'unetplusplus', 'fpn', 'deeplabv3plus', 'multiscale'
        encoder_name: Backbone encoder
        encoder_weights: Pretrained weights
        in_channels: Number of input channels
        classes: Number of output classes
        activation: Output activation
        attention_type: Attention mechanism type
    
    Returns:
        Segmentation model
    """
    models_dict = {
        'unet': UNetSegmentation,
        'unetplusplus': UNetPlusPlus,
        'fpn': FPN,
        'deeplabv3plus': DeepLabV3Plus,
        'multiscale': MultiScaleUNet
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models_dict.keys())}")
    
    model_class = models_dict[model_name]
    
    # Build kwargs based on model type
    kwargs = {
        'encoder_name': encoder_name,
        'encoder_weights': encoder_weights,
        'in_channels': in_channels,
        'classes': classes
    }
    
    if model_name in ['unet', 'unetplusplus', 'fpn', 'deeplabv3plus']:
        kwargs['activation'] = activation
    
    if model_name == 'unet' and attention_type:
        kwargs['attention_type'] = attention_type
    
    return model_class(**kwargs)


class SegmentationTrainer:
    """
    Trainer for segmentation models with mixed precision and advanced features.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module,
        learning_rate: float = 1e-4,
        use_amp: bool = True
    ):
        """
        Args:
            model: Segmentation model
            device: Device to train on
            criterion: Loss function
            learning_rate: Learning rate
            use_amp: Whether to use automatic mixed precision
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.use_amp = use_amp
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def train_epoch(self, dataloader) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for images, masks in dataloader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
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
            
            total_loss += loss.item()
        
        self.scheduler.step()
        
        return {'loss': total_loss / len(dataloader)}
    
    def validate(self, dataloader) -> dict:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, masks in dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                total_loss += loss.item()
        
        return {'loss': total_loss / len(dataloader)}
