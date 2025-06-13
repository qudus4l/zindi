#!/usr/bin/env python3
"""Script to check device availability and test CUDA/MPS setup."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.device_utils import get_device_info, get_optimal_device, get_memory_info

def main():
    """Check device availability and capabilities."""
    print("=" * 70)
    print("DEVICE AVAILABILITY CHECK")
    print("=" * 70)
    
    # Get device information
    device_info = get_device_info()
    
    print("\n1. Device Availability:")
    print(f"   - CUDA available: {device_info['cuda_available']}")
    if device_info['cuda_available']:
        print(f"   - CUDA device count: {device_info['cuda_device_count']}")
        for i, name in enumerate(device_info['cuda_device_names']):
            print(f"   - CUDA device {i}: {name}")
    print(f"   - MPS available: {device_info['mps_available']}")
    print(f"   - CPU threads: {device_info['cpu_count']}")
    
    print("\n2. PyTorch Version Information:")
    print(f"   - PyTorch version: {torch.__version__}")
    print(f"   - CUDA version (PyTorch built with): {torch.version.cuda}")
    
    print("\n3. Device Selection Test:")
    for preferred in ['auto', 'cuda', 'mps', 'cpu']:
        device = get_optimal_device(preferred)
        print(f"   - Preferred '{preferred}' -> Selected: {device}")
        
        # Get memory info if available
        memory_info = get_memory_info(device)
        if memory_info:
            print(f"     Memory: {memory_info['total']:.2f}GB total")
    
    print("\n4. Simple Computation Test:")
    device = get_optimal_device('auto')
    print(f"   Testing on device: {device}")
    
    # Create a simple tensor and perform computation
    try:
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # Time a simple operation
        import time
        start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed = time.time() - start
        
        print(f"   Matrix multiplication (1000x1000): {elapsed*1000:.2f}ms")
        print("   ✓ Device computation successful")
        
    except Exception as e:
        print(f"   ✗ Device computation failed: {e}")
    
    print("\n5. Mixed Precision Support:")
    if device.type == 'cuda':
        try:
            with torch.cuda.amp.autocast():
                x = torch.randn(100, 100, device=device)
                y = x * 2
            print("   ✓ Mixed precision (AMP) supported")
        except Exception as e:
            print(f"   ✗ Mixed precision not supported: {e}")
    else:
        print(f"   - Mixed precision not available on {device.type}")
    
    print("\n" + "=" * 70)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    if device_info['cuda_available']:
        print("✓ CUDA is available - training will be significantly faster")
        print("  Use: python scripts/train_model.py --device cuda")
    elif device_info['mps_available']:
        print("✓ MPS is available - training will be faster than CPU")
        print("  Use: python scripts/train_model.py --device mps")
    else:
        print("⚠ Only CPU available - training will be slower")
        print("  Consider using a machine with GPU for faster training")
    
    print("=" * 70)


if __name__ == "__main__":
    main() 