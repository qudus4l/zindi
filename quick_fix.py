#!/usr/bin/env python3
"""Quick fix to run training with proper paths."""

import sys
import os
import subprocess

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now run the training
subprocess.run([
    sys.executable, 
    "scripts/train_model.py",
    "--device", "cuda",
    "--epochs", "5", 
    "--batch-size", "32"
]) 