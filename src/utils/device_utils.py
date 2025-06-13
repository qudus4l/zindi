"""Device selection utilities for optimal hardware utilization.

This module provides utilities for selecting the best available device
for training and inference, with support for CUDA, MPS, and CPU.
"""

import torch
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def get_optimal_device(preferred_device: str = 'auto') -> torch.device:
    """Get the optimal device for computation.
    
    Priority order:
    1. CUDA (if available and requested)
    2. MPS (if available and requested)
    3. CPU (fallback)
    
    Args:
        preferred_device: Preferred device ('cuda', 'mps', 'cpu', or 'auto')
        
    Returns:
        torch.device: Selected device
    """
    if preferred_device == 'auto':
        # Automatic selection with priority: CUDA > MPS > CPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Auto-selected CUDA device: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Auto-selected MPS device (Apple Silicon)")
        else:
            device = torch.device('cpu')
            logger.info("Auto-selected CPU device")
    else:
        # Use specified device if available
        if preferred_device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif preferred_device == 'mps' and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using MPS device (Apple Silicon)")
        elif preferred_device == 'cpu':
            device = torch.device('cpu')
            logger.info("Using CPU device")
        else:
            # Fallback to CPU if requested device not available
            logger.warning(f"Requested device '{preferred_device}' not available, falling back to CPU")
            device = torch.device('cpu')
    
    return device


def get_device_info() -> dict:
    """Get information about available devices.
    
    Returns:
        dict: Device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_device_names': [],
        'mps_available': torch.backends.mps.is_available(),
        'cpu_count': torch.get_num_threads()
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            info['cuda_device_names'].append(torch.cuda.get_device_name(i))
    
    return info


def optimize_for_device(device: torch.device, config: dict) -> dict:
    """Optimize configuration based on device capabilities.
    
    Args:
        device: Selected device
        config: Training configuration
        
    Returns:
        dict: Optimized configuration
    """
    optimized_config = config.copy()
    
    if device.type == 'cuda':
        # CUDA optimizations
        logger.info("Applying CUDA optimizations")
        # Enable mixed precision if not already set
        if 'use_mixed_precision' in optimized_config:
            optimized_config['use_mixed_precision'] = True
        # Enable pin memory for faster data transfer
        optimized_config['pin_memory'] = True
        # Set optimal number of workers
        optimized_config['num_workers'] = 4
        
    elif device.type == 'mps':
        # MPS optimizations
        logger.info("Applying MPS optimizations")
        # Disable mixed precision (not fully supported on MPS)
        if 'use_mixed_precision' in optimized_config:
            optimized_config['use_mixed_precision'] = False
        # Disable pin memory (not needed for MPS)
        optimized_config['pin_memory'] = False
        # Set workers to 0 to avoid multiprocessing issues
        optimized_config['num_workers'] = 0
        
    else:  # CPU
        # CPU optimizations
        logger.info("Applying CPU optimizations")
        # Disable mixed precision
        if 'use_mixed_precision' in optimized_config:
            optimized_config['use_mixed_precision'] = False
        # Disable pin memory
        optimized_config['pin_memory'] = False
        # Use multiple workers for CPU
        optimized_config['num_workers'] = 2
    
    return optimized_config


def get_memory_info(device: torch.device) -> Optional[dict]:
    """Get memory information for the device.
    
    Args:
        device: Device to query
        
    Returns:
        Optional[dict]: Memory information if available
    """
    if device.type == 'cuda':
        return {
            'allocated': torch.cuda.memory_allocated(device) / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved(device) / 1024**3,  # GB
            'total': torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
        }
    return None


def clear_memory(device: torch.device) -> None:
    """Clear memory cache for the device.
    
    Args:
        device: Device to clear
    """
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA memory cache")
    elif device.type == 'mps':
        # MPS doesn't have explicit cache clearing yet
        pass 