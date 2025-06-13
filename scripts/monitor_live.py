#!/usr/bin/env python3
"""Monitor live training progress from log file."""

import time
import os
from pathlib import Path

def monitor_training_log():
    """Monitor training progress from log file."""
    log_file = Path("training_output.log")
    
    if not log_file.exists():
        print("Training log not found. Waiting...")
        return
    
    # Get file size
    file_size = log_file.stat().st_size
    print(f"Log file size: {file_size:,} bytes")
    
    # Read last 20 lines
    with open(log_file, 'r') as f:
        lines = f.readlines()
        last_lines = lines[-20:] if len(lines) > 20 else lines
        
    print("\n=== Last 20 lines of training log ===")
    for line in last_lines:
        print(line.rstrip())
    
    # Check for completion markers
    full_content = ''.join(lines)
    if "Training complete!" in full_content:
        print("\n✓ Training has completed successfully!")
    elif "ERROR:" in lines[-5:] if len(lines) > 5 else lines:
        print("\n✗ Error detected in recent output")
    else:
        print("\n⏳ Training still in progress...")
    
    # Check process
    result = os.system("ps aux | grep train_model.py | grep -v grep > /dev/null")
    if result == 0:
        print("✓ Training process is running")
    else:
        print("✗ Training process not found")

if __name__ == "__main__":
    monitor_training_log() 