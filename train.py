"""
Quick entry point for training models.
Run from project root: python train.py --help
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.main import main

if __name__ == '__main__':
    # Inject 'train' command
    if len(sys.argv) == 1:
        sys.argv.append('--help')
    main()
