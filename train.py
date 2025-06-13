#!/usr/bin/env python3
"""Launcher script for training - handles imports correctly."""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import and run the training script
if __name__ == "__main__":
    # Import here after path is set
    from scripts.train_model import main
    main() 