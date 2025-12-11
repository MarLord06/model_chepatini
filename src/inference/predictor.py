"""
Post-Processing and Prediction Pipeline for Severstal Steel Defect Detection.
Includes advanced post-processing, TTA, and submission generation.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.utils.helpers import rle_encode, post_process_mask
from src.models.classifier import DefectClassifier
from src.models.segmentation import get_segmentation_model


# ==================== POST-PROCESSOR ====================

class PostProcessor:
    """
    Advanced post-processor for segmentation masks.
    Removes false positives and refines mask quality.
    """
    
    def __init__(
        self,
        min_area: Dict[int, int] = None,
        threshold: float = 0.5,
        use_morphology: bool = True,
        kernel_size: int = 3
    ):
        """
        Args:
            min_area: Minimum area per class {class_id: min_pixels}
            threshold: Binarization threshold
            use_morphology: Apply morphological operations
            kernel_size: Kernel size for morphology
        """
        # Default minimum areas per class (based on Kaggle solutions)
        self.min_area = min_area or {
            0: 600,   # Class 1
            1: 600,   # Class 2
            2: 1000,  # Class 3
            3: 2000   # Class 4
        }
        self.threshold = threshold
        self.use_morphology = use_morphology
        self.kernel_size = kernel_size
        
        # Morphological kernels
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (kernel_size, kernel_size)
        )
    
    def process_mask(
        self,
        mask: np.ndarray,
        class_id: int,
        original_size: Tuple[int, int] = (256, 1600)
    ) -> np.ndarray:
        """
        Process a single class mask.
        
        Args:
            mask: Predicted mask (H, W) or (1, H, W)
            class_id: Class index (0-3)
            original_size: Original image size for resizing
        
        Returns:
            Processed binary mask
        """
        # Ensure 2D
        if mask.ndim == 3:
            mask = mask.squeeze()
        
        # Convert to numpy if tensor
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Binarize
        binary_mask = (mask > self.threshold).astype(np.uint8)
        
        # Resize to original size
        if binary_mask.shape != original_size:
            binary_mask = cv2.resize(
                binary_mask, 
                (original_size[1], original_size[0]),
                interpolation=cv2.INTER_LINEAR
            )
            binary_mask = (binary_mask > 0.5).astype(np.uint8)
        
        # Apply morphological operations
        if self.use_morphology:
            # Close small holes
            binary_mask = cv2.morphologyEx(
                binary_mask, cv2.MORPH_CLOSE, self.kernel
            )
            # Remove small noise
            binary_mask = cv2.morphologyEx(
                binary_mask, cv2.MORPH_OPEN, self.kernel
            )
        
        # Remove small connected components
        min_size = self.min_area.get(class_id, 600)
        binary_mask = self._filter_small_components(binary_mask, min_size)
        
        return binary_mask
    
    def _filter_small_components(
        self,
        mask: np.ndarray,
        min_size: int
    ) -> np.ndarray:
        """Remove connected components smaller than min_size."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        filtered_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                filtered_mask[labels == i] = 1
        
        return filtered_mask
    
    def process_batch(
        self,
        masks: np.ndarray,
        original_size: Tuple[int, int] = (256, 1600)
    ) -> np.ndarray:
        """
        Process batch of multi-class masks.
        
        Args:
            masks: Predicted masks (C, H, W) or (B, C, H, W)
            original_size: Original image size
        
        Returns:
            Processed masks
        """
        if masks.ndim == 4:
            # Batch processing
            batch_results = []
            for batch_mask in masks:
                result = self.process_batch(batch_mask, original_size)
                batch_results.append(result)
            return np.stack(batch_results)
        
        # Single image with multiple classes
        processed = []
        for class_id in range(masks.shape[0]):
            processed_mask = self.process_mask(
                masks[class_id], class_id, original_size
            )
            processed.append(processed_mask)
        
        return np.stack(processed)
    
    def mask_to_rle_submission(
        self,
        mask: np.ndarray,
        image_id: str
    ) -> List[Dict]:
        """
        Convert processed mask to submission format.
        
        Args:
            mask: Processed mask (C, H, W)
            image_id: Image ID for submission
        
        Returns:
            List of submission rows
        """
        submissions = []
        
        for class_id in range(mask.shape[0]):
            class_mask = mask[class_id]
            
            # Only encode if there are pixels
            if class_mask.sum() > 0:
                rle = rle_encode(class_mask)
            else:
                rle = ''
            
            submissions.append({
                'ImageId_ClassId': f"{image_id}_{class_id + 1}",
                'EncodedPixels': rle if rle else ''
            })
        
        return submissions


# ==================== TEST-TIME AUGMENTATION ====================

class TTAPredictor:
    """
    Test-Time Augmentation predictor for improved predictions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        tta_transforms: List[str] = None
    ):
        """
        Args:
            model: Segmentation model
            device: Device
            tta_transforms: List of transforms ('hflip', 'vflip', 'rotate90', etc.)
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        # Default TTA transforms
        self.tta_transforms = tta_transforms or ['original', 'hflip', 'vflip']
    
    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """
        Predict with TTA.
        
        Args:
            image: Input image (B, C, H, W)
        
        Returns:
            Averaged prediction
        """
        predictions = []
        
        for transform_name in self.tta_transforms:
            # Apply transform
            transformed = self._apply_transform(image, transform_name)
            
            # Predict
            pred = self.model(transformed.to(self.device))
            pred = torch.sigmoid(pred)
            
            # Reverse transform
            pred = self._reverse_transform(pred, transform_name)
            
            predictions.append(pred)
        
        # Average predictions
        return torch.stack(predictions).mean(dim=0)
    
    def _apply_transform(
        self,
        image: torch.Tensor,
        transform_name: str
    ) -> torch.Tensor:
        """Apply TTA transform."""
        if transform_name == 'original':
            return image
        elif transform_name == 'hflip':
            return torch.flip(image, dims=[3])
        elif transform_name == 'vflip':
            return torch.flip(image, dims=[2])
        elif transform_name == 'hvflip':
            return torch.flip(image, dims=[2, 3])
        elif transform_name == 'rotate90':
            return torch.rot90(image, k=1, dims=[2, 3])
        elif transform_name == 'rotate180':
            return torch.rot90(image, k=2, dims=[2, 3])
        elif transform_name == 'rotate270':
            return torch.rot90(image, k=3, dims=[2, 3])
        else:
            return image
    
    def _reverse_transform(
        self,
        pred: torch.Tensor,
        transform_name: str
    ) -> torch.Tensor:
        """Reverse TTA transform on prediction."""
        if transform_name == 'original':
            return pred
        elif transform_name == 'hflip':
            return torch.flip(pred, dims=[3])
        elif transform_name == 'vflip':
            return torch.flip(pred, dims=[2])
        elif transform_name == 'hvflip':
            return torch.flip(pred, dims=[2, 3])
        elif transform_name == 'rotate90':
            return torch.rot90(pred, k=-1, dims=[2, 3])
        elif transform_name == 'rotate180':
            return torch.rot90(pred, k=-2, dims=[2, 3])
        elif transform_name == 'rotate270':
            return torch.rot90(pred, k=-3, dims=[2, 3])
        else:
            return pred


# ==================== ENSEMBLE INFERENCE ====================

class EnsembleInference:
    """
    Complete inference pipeline with ensemble models.
    """
    
    def __init__(
        self,
        classifier: nn.Module,
        segmentation_models: List[nn.Module],
        post_processor: PostProcessor,
        device: torch.device,
        classifier_threshold: float = 0.5,
        segmentation_threshold: float = 0.5,
        model_weights: Optional[List[float]] = None,
        use_tta: bool = True
    ):
        """
        Args:
            classifier: Binary classifier model
            segmentation_models: List of segmentation models
            post_processor: Post-processor instance
            device: Device
            classifier_threshold: Threshold for classifier
            segmentation_threshold: Threshold for segmentation
            model_weights: Weights for ensemble averaging
            use_tta: Whether to use TTA
        """
        self.classifier = classifier.to(device).eval()
        self.segmentation_models = [m.to(device).eval() for m in segmentation_models]
        self.post_processor = post_processor
        self.device = device
        self.classifier_threshold = classifier_threshold
        self.segmentation_threshold = segmentation_threshold
        self.use_tta = use_tta
        
        # Normalize weights
        if model_weights is None:
            n_models = len(segmentation_models)
            self.model_weights = [1.0 / n_models] * n_models
        else:
            total = sum(model_weights)
            self.model_weights = [w / total for w in model_weights]
        
        # TTA predictors
        if use_tta:
            self.tta_predictors = [
                TTAPredictor(model, device) for model in self.segmentation_models
            ]
    
    @torch.no_grad()
    def predict_single(
        self,
        image: torch.Tensor,
        original_size: Tuple[int, int] = (256, 1600)
    ) -> Tuple[bool, np.ndarray]:
        """
        Predict for a single image.
        
        Args:
            image: Input image (1, 3, H, W) or (3, H, W)
            original_size: Original image size
        
        Returns:
            Tuple of (has_defect, processed_mask)
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        # Stage 1: Binary classification
        classifier_out = self.classifier(image)
        has_defect_prob = torch.sigmoid(classifier_out).item()
        
        if has_defect_prob < self.classifier_threshold:
            # No defect detected
            return False, np.zeros((4, *original_size), dtype=np.uint8)
        
        # Stage 2: Segmentation ensemble
        predictions = []
        
        for i, model in enumerate(self.segmentation_models):
            if self.use_tta:
                pred = self.tta_predictors[i].predict(image)
            else:
                pred = torch.sigmoid(model(image))
            
            predictions.append(pred * self.model_weights[i])
        
        # Ensemble average
        ensemble_pred = torch.stack(predictions).sum(dim=0)
        
        # Convert to numpy
        mask = ensemble_pred.squeeze(0).cpu().numpy()
        
        # Post-process
        processed_mask = self.post_processor.process_batch(mask, original_size)
        
        # Check if any defects remain after post-processing
        has_defect = processed_mask.sum() > 0
        
        return has_defect, processed_mask
    
    def predict_dataloader(
        self,
        dataloader: DataLoader,
        original_size: Tuple[int, int] = (256, 1600)
    ) -> List[Dict]:
        """
        Predict for entire dataloader.
        
        Args:
            dataloader: Data loader with images
            original_size: Original image size
        
        Returns:
            List of predictions
        """
        all_predictions = []
        
        for batch in tqdm(dataloader, desc='Predicting'):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
                image_ids = batch[1] if len(batch) > 1 else None
            else:
                images = batch
                image_ids = None
            
            for i in range(images.size(0)):
                has_defect, mask = self.predict_single(images[i], original_size)
                
                prediction = {
                    'has_defect': has_defect,
                    'mask': mask
                }
                
                if image_ids is not None:
                    prediction['image_id'] = image_ids[i]
                
                all_predictions.append(prediction)
        
        return all_predictions


# ==================== SUBMISSION GENERATION ====================

class SubmissionGenerator:
    """
    Generate Kaggle submission file.
    """
    
    def __init__(
        self,
        ensemble_inference: EnsembleInference,
        post_processor: PostProcessor
    ):
        self.inference = ensemble_inference
        self.post_processor = post_processor
    
    def generate(
        self,
        test_loader: DataLoader,
        image_ids: List[str],
        output_path: str,
        original_size: Tuple[int, int] = (256, 1600)
    ) -> pd.DataFrame:
        """
        Generate submission file.
        
        Args:
            test_loader: Test data loader
            image_ids: List of image IDs
            output_path: Path for output CSV
            original_size: Original image size
        
        Returns:
            Submission DataFrame
        """
        submissions = []
        idx = 0
        
        for batch in tqdm(test_loader, desc='Generating submission'):
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            
            for i in range(images.size(0)):
                has_defect, mask = self.inference.predict_single(
                    images[i], original_size
                )
                
                image_id = image_ids[idx]
                
                # Generate RLE for each class
                for class_id in range(4):
                    class_mask = mask[class_id]
                    
                    if has_defect and class_mask.sum() > 0:
                        rle = rle_encode(class_mask)
                    else:
                        rle = ''
                    
                    submissions.append({
                        'ImageId_ClassId': f"{image_id}_{class_id + 1}",
                        'EncodedPixels': rle
                    })
                
                idx += 1
        
        # Create DataFrame
        df = pd.DataFrame(submissions)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Submission saved to: {output_path}")
        
        # Statistics
        n_with_defects = (df['EncodedPixels'] != '').sum()
        print(f"Total predictions: {len(df)}")
        print(f"Predictions with defects: {n_with_defects} ({n_with_defects/len(df)*100:.1f}%)")
        
        return df


# ==================== TEST DATASET ====================

class TestDataset(Dataset):
    """Simple test dataset for inference."""
    
    def __init__(
        self,
        image_dir: str,
        image_ids: List[str],
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (256, 512)
    ):
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.image_size = image_size
        
        self.transform = transform or A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, image_id)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        augmented = self.transform(image=image)
        image = augmented['image']
        
        return image, image_id


# ==================== MAIN PREDICTION FUNCTION ====================

def create_submission(
    classifier_path: str,
    segmentation_paths: List[str],
    test_image_dir: str,
    submission_csv_path: str,
    device: str = 'auto',
    batch_size: int = 8,
    use_tta: bool = True
):
    """
    Create submission file for Kaggle.
    
    Args:
        classifier_path: Path to classifier model
        segmentation_paths: List of paths to segmentation models
        test_image_dir: Directory with test images
        submission_csv_path: Path for output submission
        device: Device
        batch_size: Batch size
        use_tta: Use test-time augmentation
    """
    from src.models.classifier import DefectClassifier
    from src.models.segmentation import get_segmentation_model
    
    # Setup device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Load classifier
    print("Loading classifier...")
    classifier = DefectClassifier()
    checkpoint = torch.load(classifier_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        classifier.load_state_dict(checkpoint)
    classifier.to(device)
    classifier.eval()
    
    # Load segmentation models
    print("Loading segmentation models...")
    segmentation_models = []
    for path in segmentation_paths:
        # Extract model info from path
        path_str = str(path).lower()
        if 'unetplusplus' in path_str:
            model_name = 'unetplusplus'
        elif 'unet' in path_str:
            model_name = 'unet'
        elif 'fpn' in path_str:
            model_name = 'fpn'
        elif 'deeplabv3' in path_str:
            model_name = 'deeplabv3plus'
        else:
            model_name = 'unet'  # default
        
        # Extract encoder from path
        if 'efficientnet-b4' in path_str or 'efficientnet_b4' in path_str:
            encoder_name = 'efficientnet-b4'
        elif 'se_resnext50' in path_str:
            encoder_name = 'se_resnext50_32x4d'
        elif 'resnet50' in path_str:
            encoder_name = 'resnet50'
        else:
            encoder_name = 'efficientnet-b4'  # default
        
        print(f"Loading {model_name} with {encoder_name} encoder from {path}")
        model = get_segmentation_model(
            model_name=model_name,
            encoder_name=encoder_name,
            encoder_weights=None  # We'll load our trained weights
        )
        checkpoint = torch.load(path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        segmentation_models.append(model)
    
    # Create post-processor
    post_processor = PostProcessor()
    
    # Create ensemble inference
    ensemble = EnsembleInference(
        classifier=classifier,
        segmentation_models=segmentation_models,
        post_processor=post_processor,
        device=device,
        use_tta=use_tta
    )
    
    # Get test image IDs
    test_image_ids = [f for f in os.listdir(test_image_dir) if f.endswith('.jpg')]
    print(f"Found {len(test_image_ids)} test images")
    
    # Create test dataset and loader
    test_dataset = TestDataset(test_image_dir, test_image_ids)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Generate submission
    generator = SubmissionGenerator(ensemble, post_processor)
    submission_df = generator.generate(
        test_loader=test_loader,
        image_ids=test_image_ids,
        output_path=submission_csv_path
    )
    
    return submission_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', type=str, required=True)
    parser.add_argument('--segmentation', type=str, nargs='+', required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='submission.csv')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--no_tta', action='store_true')
    
    args = parser.parse_args()
    
    create_submission(
        classifier_path=args.classifier,
        segmentation_paths=args.segmentation,
        test_image_dir=args.test_dir,
        submission_csv_path=args.output,
        device=args.device,
        batch_size=args.batch_size,
        use_tta=not args.no_tta
    )
