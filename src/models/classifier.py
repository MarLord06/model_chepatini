"""
Binary Classifier to detect presence of defects in steel images.
This model acts as a gatekeeper before running expensive segmentation.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class DefectClassifier(nn.Module):
    """
    Binary classifier based on EfficientNet-B4 backbone.
    Predicts whether an image contains any defects.
    """
    
    def __init__(self, backbone: str = 'efficientnet_b4', pretrained: bool = True):
        """
        Args:
            backbone: Backbone architecture ('efficientnet_b4', 'resnet50', 'efficientnet_b3')
            pretrained: Whether to use ImageNet pretrained weights
        """
        super(DefectClassifier, self).__init__()
        
        self.backbone_name = backbone
        
        if 'efficientnet' in backbone:
            if backbone == 'efficientnet_b4':
                self.backbone = models.efficientnet_b4(pretrained=pretrained)
                num_features = self.backbone.classifier[1].in_features
                self.backbone.classifier = nn.Identity()
            elif backbone == 'efficientnet_b3':
                self.backbone = models.efficientnet_b3(pretrained=pretrained)
                num_features = self.backbone.classifier[1].in_features
                self.backbone.classifier = nn.Identity()
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")
        
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
        
        Returns:
            Logits of shape (batch_size, 1)
        """
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict probability of defect presence.
        
        Args:
            x: Input tensor
        
        Returns:
            Probabilities in range [0, 1]
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)


class MultiHeadClassifier(nn.Module):
    """
    Multi-head classifier that predicts:
    1. Whether image has any defects (binary)
    2. Which specific defect classes are present (multi-label)
    """
    
    def __init__(self, backbone: str = 'efficientnet_b4', num_classes: int = 4, pretrained: bool = True):
        """
        Args:
            backbone: Backbone architecture
            num_classes: Number of defect classes (4 for Severstal)
            pretrained: Whether to use pretrained weights
        """
        super(MultiHeadClassifier, self).__init__()
        
        if 'efficientnet' in backbone:
            if backbone == 'efficientnet_b4':
                self.backbone = models.efficientnet_b4(pretrained=pretrained)
                num_features = self.backbone.classifier[1].in_features
                self.backbone.classifier = nn.Identity()
            elif backbone == 'efficientnet_b3':
                self.backbone = models.efficientnet_b3(pretrained=pretrained)
                num_features = self.backbone.classifier[1].in_features
                self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Shared feature extraction
        self.shared_fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )
        
        # Binary head: has defect or not
        self.binary_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        # Multi-label head: which defect classes
        self.multilabel_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
        
        Returns:
            Tuple of (binary_logits, multilabel_logits)
        """
        features = self.backbone(x)
        shared = self.shared_fc(features)
        
        binary_out = self.binary_head(shared)
        multilabel_out = self.multilabel_head(shared)
        
        return binary_out, multilabel_out
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> tuple:
        """
        Predict with threshold.
        
        Args:
            x: Input tensor
            threshold: Classification threshold
        
        Returns:
            Tuple of (has_defect_prob, class_probs)
        """
        binary_logits, multilabel_logits = self.forward(x)
        
        binary_prob = torch.sigmoid(binary_logits)
        multilabel_prob = torch.sigmoid(multilabel_logits)
        
        return binary_prob, multilabel_prob


def get_classifier(model_type: str = 'binary', **kwargs) -> nn.Module:
    """
    Factory function to create classifier models.
    
    Args:
        model_type: 'binary' or 'multihead'
        **kwargs: Additional arguments for the model
    
    Returns:
        Classifier model
    """
    if model_type == 'binary':
        return DefectClassifier(**kwargs)
    elif model_type == 'multihead':
        return MultiHeadClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class ClassifierTrainer:
    """
    Trainer class for binary classifier.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-4,
        pos_weight: Optional[float] = None
    ):
        """
        Args:
            model: Classifier model
            device: Device to train on
            learning_rate: Learning rate
            pos_weight: Weight for positive class (to handle imbalance)
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # BCE loss with optional class weighting
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight]).to(device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
    
    def train_epoch(self, dataloader) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device).float().unsqueeze(1)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.sigmoid(outputs) > 0.5
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total
        }
    
    def validate(self, dataloader) -> dict:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                predictions = torch.sigmoid(outputs) > 0.5
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        self.scheduler.step(avg_loss)
        
        return {
            'loss': avg_loss,
            'accuracy': correct / total
        }
