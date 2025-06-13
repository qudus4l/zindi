#!/usr/bin/env python3
"""Monitor training progress by checking output files."""

import os
import time
from pathlib import Path
import json

def monitor_training():
    """Monitor training progress."""
    print("Monitoring training progress...")
    print("=" * 60)
    
    # Check for checkpoint directories
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("epoch_*"))
        if checkpoints:
            print(f"Found {len(checkpoints)} checkpoint(s):")
            for cp in sorted(checkpoints):
                print(f"  - {cp.name}")
        else:
            print("No checkpoints found yet.")
    else:
        print("Checkpoint directory not created yet.")
    
    # Check for training results
    results_dir = Path("training_results")
    if results_dir.exists():
        reports = list(results_dir.glob("training_report_*.json"))
        if reports:
            print(f"\nFound {len(reports)} training report(s):")
            for report in sorted(reports):
                print(f"  - {report.name}")
                # Try to read and show some info
                try:
                    with open(report, 'r') as f:
                        data = json.load(f)
                        if 'best_rouge_score' in data:
                            print(f"    Best ROUGE-1 F1: {data['best_rouge_score']:.4f}")
                except:
                    pass
    
    # Check process status
    print("\nChecking if training is still running...")
    result = os.system("ps aux | grep train_model.py | grep -v grep > /dev/null")
    if result == 0:
        print("✓ Training process is still running")
    else:
        print("✗ Training process has completed or stopped")
    
    print("=" * 60)

if __name__ == "__main__":
    monitor_training() 