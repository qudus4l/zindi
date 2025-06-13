#!/usr/bin/env python3
"""Run the complete training pipeline with CUDA."""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and print output."""
    print(f"\n>>> Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error running command: {cmd}")
        sys.exit(1)

def main():
    print("="*60)
    print("CLINICAL DECISION SUPPORT MODEL - FULL TRAINING PIPELINE")
    print("="*60)
    
    # Step 1: Check CUDA
    print("\n[1/2] Checking CUDA availability...")
    run_command("python scripts/check_device.py")
    
    # Step 2: Run training
    print("\n[2/2] Starting model training (5 epochs with CUDA)...")
    run_command("python scripts/train_model.py --device cuda --epochs 5 --batch-size 16")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nResults saved to:")
    print("  - Model checkpoints: ./checkpoints/")
    print("  - Training reports: ./training_results/")
    print("  - Logs: ./training.log")
    print("\nTo run inference on test data:")
    print("  python scripts/inference.py --model-path checkpoints/epoch_X/")

if __name__ == "__main__":
    main() 