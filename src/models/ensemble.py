"""
Ensemble Model combining Binary Classifier and Segmentation Models.
Two-stage approach: First classify if defects exist, then segment if needed.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
import torch.nn.functional as F


class EnsemblePredictor:
    """
    Ensemble predictor that combines:
    1. Binary classifier to detect defect presence
    2. Multiple segmentation models for defect localization
    3. Post-processing to refine predictions
    """
    
    def __init__(
        self,
        classifier: nn.Module,
        segmentation_models: List[nn.Module],
        device: torch.device,
        classifier_threshold: float = 0.5,
        segmentation_threshold: float = 0.5,
        min_defect_size: int = 800,
        weights: Optional[List[float]] = None
    ):
        """
        Args:
            classifier: Binary classifier model
            segmentation_models: List of segmentation models
            device: Device to run inference on
            classifier_threshold: Threshold for classifier prediction
            segmentation_threshold: Threshold for segmentation
            min_defect_size: Minimum defect size (pixels) to keep
            weights: Optional weights for ensemble averaging
        """
        self.classifier = classifier.to(device).eval()
        self.segmentation_models = [model.to(device).eval() for model in segmentation_models]
        self.device = device
        self.classifier_threshold = classifier_threshold
        self.segmentation_threshold = segmentation_threshold
        self.min_defect_size = min_defect_size
        
        # Ensemble weights (uniform if not provided)
        if weights is None:
            self.weights = [1.0 / len(segmentation_models)] * len(segmentation_models)
        else:
            assert len(weights) == len(segmentation_models), "Weights must match number of models"
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        apply_tta: bool = False
    ) -> Tuple[bool, Optional[torch.Tensor]]:
        """
        Two-stage prediction pipeline.
        
        Args:
            image: Input image tensor (1, 3, H, W) or (3, H, W)
            apply_tta: Whether to apply test-time augmentation
        
        Returns:
            Tuple of (has_defect, segmentation_mask)
            - has_defect: Boolean indicating if defects are present
            - segmentation_mask: Tensor of shape (4, H, W) or None if no defects
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        # Stage 1: Binary classification
        has_defect = self._classify(image)
        
        if not has_defect:
            # No defects detected, return empty mask
            return False, None
        
        # Stage 2: Segmentation (only if defects detected)
        if apply_tta:
            mask = self._segment_with_tta(image)
        else:
            mask = self._segment(image)
        
        # Post-processing
        mask = self._post_process(mask)
        
        # Check if post-processing removed all defects
        if mask.sum() == 0:
            return False, None
        
        return True, mask
    
    def _classify(self, image: torch.Tensor) -> bool:
        """Run binary classification."""
        logits = self.classifier(image)
        prob = torch.sigmoid(logits).item()
        return prob >= self.classifier_threshold
    
    def _segment(self, image: torch.Tensor) -> torch.Tensor:
        """Run ensemble segmentation."""
        predictions = []
        
        for model, weight in zip(self.segmentation_models, self.weights):
            output = model(image)
            # Apply sigmoid if needed
            if output.min() < 0 or output.max() > 1:
                output = torch.sigmoid(output)
            predictions.append(output * weight)
        
        # Weighted average
        ensemble_pred = torch.stack(predictions).sum(dim=0)
        return ensemble_pred.squeeze(0)  # Remove batch dimension
    
    def _segment_with_tta(self, image: torch.Tensor) -> torch.Tensor:
        """Run segmentation with test-time augmentation."""
        predictions = []
        
        # Original
        pred = self._segment(image)
        predictions.append(pred)
        
        # Horizontal flip
        pred = self._segment(torch.flip(image, dims=[3]))
        pred = torch.flip(pred, dims=[2])
        predictions.append(pred)
        
        # Vertical flip
        pred = self._segment(torch.flip(image, dims=[2]))
        pred = torch.flip(pred, dims=[1])
        predictions.append(pred)
        
        # Both flips
        pred = self._segment(torch.flip(image, dims=[2, 3]))
        pred = torch.flip(pred, dims=[1, 2])
        predictions.append(pred)
        
        # Average all TTA predictions
        return torch.stack(predictions).mean(dim=0)
    
    def _post_process(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Post-process segmentation mask:
        - Apply threshold
        - Remove small connected components
        - Apply morphological operations
        """
        import cv2
        
        # Convert to numpy for processing
        mask_np = mask.cpu().numpy()
        processed_masks = []
        
        for class_idx in range(mask_np.shape[0]):
            class_mask = mask_np[class_idx]
            
            # Binarize
            binary_mask = (class_mask > self.segmentation_threshold).astype(np.uint8)
            
            # Remove small components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary_mask, connectivity=8
            )
            
            filtered_mask = np.zeros_like(binary_mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= self.min_defect_size:
                    filtered_mask[labels == i] = 1
            
            # Morphological operations to smooth
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)
            filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, kernel)
            
            processed_masks.append(filtered_mask)
        
        return torch.tensor(np.stack(processed_masks), dtype=torch.float32)
    
    def predict_batch(
        self,
        images: torch.Tensor,
        apply_tta: bool = False
    ) -> List[Tuple[bool, Optional[torch.Tensor]]]:
        """
        Predict for a batch of images.
        
        Args:
            images: Batch of images (B, 3, H, W)
            apply_tta: Whether to apply TTA
        
        Returns:
            List of (has_defect, mask) tuples
        """
        results = []
        for i in range(images.size(0)):
            result = self.predict(images[i], apply_tta=apply_tta)
            results.append(result)
        return results


class WeightedEnsemble:
    """
    Ensemble that learns optimal weights for combining models.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        device: torch.device,
        num_classes: int = 4
    ):
        """
        Args:
            models: List of segmentation models
            device: Device
            num_classes: Number of classes
        """
        self.models = [model.to(device).eval() for model in models]
        self.device = device
        self.num_classes = num_classes
        
        # Learnable weights (initialized uniformly)
        self.weights = nn.Parameter(
            torch.ones(len(models), device=device) / len(models)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with weighted ensemble.
        
        Args:
            x: Input image
        
        Returns:
            Weighted ensemble prediction
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                output = model(x)
                if output.min() < 0 or output.max() > 1:
                    output = torch.sigmoid(output)
                predictions.append(output)
        
        # Apply softmax to weights for normalization
        normalized_weights = F.softmax(self.weights, dim=0)
        
        # Weighted sum
        ensemble_output = sum(
            pred * weight for pred, weight in zip(predictions, normalized_weights)
        )
        
        return ensemble_output
    
    def optimize_weights(
        self,
        val_loader,
        criterion: nn.Module,
        num_epochs: int = 10,
        lr: float = 0.01
    ):
        """
        Optimize ensemble weights on validation set.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            num_epochs: Number of optimization epochs
            lr: Learning rate
        """
        optimizer = torch.optim.Adam([self.weights], lr=lr)
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.forward(images)
                loss = criterion(outputs, masks)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(val_loader):.4f}")
            print(f"Weights: {F.softmax(self.weights, dim=0).detach().cpu().numpy()}")


class CascadeEnsemble:
    """
    Cascade ensemble where models are applied sequentially,
    each refining the prediction of the previous one.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        device: torch.device
    ):
        """
        Args:
            models: List of models in cascade order
            device: Device
        """
        self.models = [model.to(device).eval() for model in models]
        self.device = device
    
    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """
        Cascade prediction where each model refines previous output.
        
        Args:
            image: Input image
        
        Returns:
            Final refined prediction
        """
        current_pred = None
        
        for i, model in enumerate(self.models):
            if i == 0:
                # First model processes raw image
                output = model(image)
            else:
                # Subsequent models use concatenation of image and previous prediction
                if output.min() < 0 or output.max() > 1:
                    output = torch.sigmoid(output)
                
                # Concatenate image and previous prediction
                model_input = torch.cat([image, output], dim=1)
                output = model(model_input)
        
        return output


def create_ensemble(
    classifier_path: str,
    segmentation_paths: List[str],
    device: torch.device,
    ensemble_type: str = 'standard',
    **kwargs
) -> EnsemblePredictor:
    """
    Factory function to create ensemble predictor.
    
    Args:
        classifier_path: Path to trained classifier
        segmentation_paths: List of paths to trained segmentation models
        device: Device
        ensemble_type: Type of ensemble ('standard', 'weighted', 'cascade')
        **kwargs: Additional arguments for ensemble
    
    Returns:
        Ensemble predictor
    """
    from classifier import DefectClassifier
    from segmentation_models import get_segmentation_model
    
    # Load classifier
    classifier = DefectClassifier(**kwargs.get('classifier_kwargs', {}))
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    
    # Load segmentation models
    segmentation_models = []
    for path in segmentation_paths:
        model = get_segmentation_model(**kwargs.get('segmentation_kwargs', {}))
        model.load_state_dict(torch.load(path, map_location=device))
        segmentation_models.append(model)
    
    return EnsemblePredictor(
        classifier=classifier,
        segmentation_models=segmentation_models,
        device=device,
        **kwargs.get('ensemble_kwargs', {})
    )
