"""
Quick entry point for making predictions.
Run from project root: python predict.py --help
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import argparse
from src.inference.predictor import create_submission
from src.inference.visualizer import visualize_predictions


def main():
    parser = argparse.ArgumentParser(description='Severstal Steel Defect Detection - Prediction')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Submission command
    submit_parser = subparsers.add_parser('submit', help='Generate submission CSV')
    submit_parser.add_argument('--classifier', type=str, required=True, help='Path to classifier model')
    submit_parser.add_argument('--segmentation', type=str, nargs='+', required=True, help='Paths to segmentation models')
    submit_parser.add_argument('--test_dir', type=str, default='data/test_images', help='Test images directory')
    submit_parser.add_argument('--output', type=str, default='submission.csv', help='Output path')
    submit_parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    submit_parser.add_argument('--no_tta', action='store_true', help='Disable TTA')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize predictions')
    viz_parser.add_argument('--submission', type=str, default='submission.csv', help='Submission CSV path')
    viz_parser.add_argument('--test_dir', type=str, default='data/test_images', help='Test images directory')
    viz_parser.add_argument('--num_images', type=int, default=10, help='Number of images')
    viz_parser.add_argument('--save_dir', type=str, default='visualizations', help='Save directory')
    
    args = parser.parse_args()
    
    if args.command == 'submit':
        create_submission(
            classifier_path=args.classifier,
            segmentation_paths=args.segmentation,
            test_image_dir=args.test_dir,
            submission_csv_path=args.output,
            batch_size=args.batch_size,
            use_tta=not args.no_tta
        )
    elif args.command == 'visualize':
        visualize_predictions(
            submission_path=args.submission,
            test_image_dir=args.test_dir,
            num_images=args.num_images,
            save_dir=args.save_dir
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
