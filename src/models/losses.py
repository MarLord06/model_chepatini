"""
Loss functions for Steel Defect Detection.
Includes various loss functions for both classification and segmentation tasks.
PyTorch implementation with advanced losses from top Kaggle solutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ==================== DICE LOSS ====================

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    Measures overlap between prediction and ground truth.
    """
    
    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: 'mean', 'sum', or 'none'
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Predicted masks (B, C, H, W) - logits or probabilities
            y_true: Ground truth masks (B, C, H, W)
        
        Returns:
            Dice loss value
        """
        # Apply sigmoid if logits
        if y_pred.min() < 0 or y_pred.max() > 1:
            y_pred = torch.sigmoid(y_pred)
        
        # Flatten spatial dimensions
        y_pred_flat = y_pred.view(y_pred.size(0), y_pred.size(1), -1)
        y_true_flat = y_true.view(y_true.size(0), y_true.size(1), -1)
        
        intersection = (y_pred_flat * y_true_flat).sum(dim=2)
        union = y_pred_flat.sum(dim=2) + y_true_flat.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        return dice_loss


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss - differentiable version for training.
    """
    
    def __init__(self, smooth: float = 1.0, p: float = 2):
        """
        Args:
            smooth: Smoothing factor
            p: Power for denominator (2 for standard Dice)
        """
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.min() < 0 or y_pred.max() > 1:
            y_pred = torch.sigmoid(y_pred)
        
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)
        
        intersection = (y_pred * y_true).sum()
        
        if self.p == 2:
            denominator = (y_pred ** 2).sum() + (y_true ** 2).sum()
        else:
            denominator = y_pred.sum() + y_true.sum()
        
        return 1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth)


# ==================== FOCAL LOSS ====================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Down-weights easy examples and focuses on hard examples.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Predictions (logits)
            y_true: Ground truth labels
        
        Returns:
            Focal loss value
        """
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
        
        # Get probabilities
        p = torch.sigmoid(y_pred)
        p_t = p * y_true + (1 - p) * (1 - y_true)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weight
        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        
        focal_loss = alpha_t * focal_weight * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for binary classification.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(y_pred)
        
        ce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
        p_t = p * y_true + (1 - p) * (1 - y_true)
        
        focal_loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


# ==================== LOVASZ LOSS ====================

def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute gradient of the Lovasz extension."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Lovasz hinge loss for binary segmentation."""
    if len(labels) == 0:
        return logits.sum() * 0.0
    
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge loss for binary segmentation.
    Better for IoU optimization than cross-entropy.
    """
    
    def __init__(self, per_image: bool = True):
        """
        Args:
            per_image: Compute loss per image then average
        """
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Predictions (logits)
            y_true: Binary labels
        
        Returns:
            Lovasz hinge loss
        """
        if self.per_image:
            losses = []
            for pred, true in zip(y_pred, y_true):
                loss = lovasz_hinge_flat(
                    pred.view(-1),
                    true.view(-1)
                )
                losses.append(loss)
            return torch.stack(losses).mean()
        else:
            return lovasz_hinge_flat(
                y_pred.view(-1),
                y_true.view(-1)
            )


# ==================== TVERSKY LOSS ====================

class TverskyLoss(nn.Module):
    """
    Tversky loss for imbalanced data.
    Generalizes Dice loss with controllable FP/FN weighting.
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-6
    ):
        """
        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing factor
        
        Note: alpha + beta = 1 gives Dice, alpha = beta = 0.5 gives F1
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.min() < 0 or y_pred.max() > 1:
            y_pred = torch.sigmoid(y_pred)
        
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)
        
        tp = (y_pred_flat * y_true_flat).sum()
        fp = ((1 - y_true_flat) * y_pred_flat).sum()
        fn = (y_true_flat * (1 - y_pred_flat)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1.0 - tversky


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - combines Focal and Tversky.
    Excellent for highly imbalanced segmentation.
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = 1e-6
    ):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.min() < 0 or y_pred.max() > 1:
            y_pred = torch.sigmoid(y_pred)
        
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)
        
        tp = (y_pred_flat * y_true_flat).sum()
        fp = ((1 - y_true_flat) * y_pred_flat).sum()
        fn = (y_true_flat * (1 - y_pred_flat)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return (1.0 - tversky) ** self.gamma


# ==================== COMBINED LOSSES ====================

class BCEDiceLoss(nn.Module):
    """
    Combined BCE and Dice loss.
    BCE provides per-pixel supervision, Dice optimizes overlap.
    """
    
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6
    ):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(y_pred, y_true)
        dice_loss = self.dice(y_pred, y_true)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class BCEFocalDiceLoss(nn.Module):
    """
    Combined BCE, Focal, and Dice loss for robust training.
    """
    
    def __init__(
        self,
        bce_weight: float = 0.2,
        focal_weight: float = 0.3,
        dice_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super(BCEFocalDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss()
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(y_pred, y_true)
        focal_loss = self.focal(y_pred, y_true)
        dice_loss = self.dice(y_pred, y_true)
        
        return (
            self.bce_weight * bce_loss +
            self.focal_weight * focal_loss +
            self.dice_weight * dice_loss
        )


class SymmetricLovaszLoss(nn.Module):
    """
    Symmetric Lovasz loss - handles both positive and negative classes.
    Used by top solutions in Severstal competition.
    """
    
    def __init__(self, per_image: bool = True):
        super(SymmetricLovaszLoss, self).__init__()
        self.per_image = per_image
        self.lovasz = LovaszHingeLoss(per_image=per_image)
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            self.lovasz(y_pred, y_true) +
            self.lovasz(-y_pred, 1.0 - y_true)
        )


# ==================== MULTI-CLASS LOSSES ====================

class MultiClassDiceLoss(nn.Module):
    """
    Dice loss for multi-class segmentation.
    Computes Dice per class and averages.
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        smooth: float = 1e-6,
        class_weights: Optional[torch.Tensor] = None
    ):
        super(MultiClassDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.class_weights = class_weights
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.min() < 0 or y_pred.max() > 1:
            y_pred = torch.sigmoid(y_pred)
        
        total_loss = 0.0
        
        for c in range(self.num_classes):
            pred_c = y_pred[:, c, :, :].contiguous().view(-1)
            true_c = y_true[:, c, :, :].contiguous().view(-1)
            
            intersection = (pred_c * true_c).sum()
            union = pred_c.sum() + true_c.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1.0 - dice
            
            if self.class_weights is not None:
                dice_loss = dice_loss * self.class_weights[c]
            
            total_loss += dice_loss
        
        return total_loss / self.num_classes


# ==================== FACTORY FUNCTION ====================

def get_loss(
    loss_name: str,
    **kwargs
) -> nn.Module:
    """
    Factory function to get loss by name.
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Additional arguments
    
    Returns:
        Loss module
    """
    losses = {
        'dice': DiceLoss,
        'soft_dice': SoftDiceLoss,
        'focal': FocalLoss,
        'binary_focal': BinaryFocalLoss,
        'lovasz': LovaszHingeLoss,
        'tversky': TverskyLoss,
        'focal_tversky': FocalTverskyLoss,
        'bce_dice': BCEDiceLoss,
        'bce_focal_dice': BCEFocalDiceLoss,
        'symmetric_lovasz': SymmetricLovaszLoss,
        'multiclass_dice': MultiClassDiceLoss,
        'bce': nn.BCEWithLogitsLoss
    }
    
    if loss_name.lower() not in losses:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(losses.keys())}")
    
    return losses[loss_name.lower()](**kwargs)
