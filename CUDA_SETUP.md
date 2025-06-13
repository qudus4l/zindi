# CUDA Support for Clinical Decision Support Model

This document describes how to use CUDA acceleration for training and inference.

## Device Support

The project now supports multiple compute devices with automatic selection:

- **CUDA**: NVIDIA GPUs (highest priority)
- **MPS**: Apple Silicon GPUs
- **CPU**: Fallback option

## Checking Device Availability

Run the device check script to see what's available on your system:

```bash
python scripts/check_device.py
```

This will show:
- Available devices (CUDA, MPS, CPU)
- PyTorch and CUDA versions
- Device memory information
- Performance test results
- Recommendations for your system

## Using CUDA for Training

### Automatic Device Selection (Recommended)

By default, the training script will automatically select the best available device:

```bash
python scripts/train_model.py
```

Priority order: CUDA → MPS → CPU

### Manual Device Selection

You can explicitly specify which device to use:

```bash
# Force CUDA usage
python scripts/train_model.py --device cuda

# Force CPU usage (useful for debugging)
python scripts/train_model.py --device cpu

# Use Apple Silicon GPU
python scripts/train_model.py --device mps
```

### Configuration File

You can also set the device in the configuration:

```python
config.training.device = 'cuda'  # or 'mps', 'cpu', 'auto'
```

## Performance Benefits

### CUDA Advantages

When using CUDA, the following optimizations are automatically enabled:

1. **Mixed Precision Training**: Uses FP16 for faster computation
2. **Pin Memory**: Faster CPU-GPU data transfer
3. **Multi-worker Data Loading**: Parallel data preprocessing
4. **Optimized Batch Sizes**: Can use larger batches

### Expected Speedups

Typical training speedups compared to CPU:

- **CUDA (RTX 3090)**: 10-15x faster
- **CUDA (V100)**: 15-20x faster
- **MPS (M1 Pro)**: 3-5x faster

## Memory Management

The system includes automatic memory management:

- Periodic cache clearing during training
- Memory usage monitoring
- Automatic batch size adjustment if OOM occurs

## Inference with CUDA

For inference/prediction:

```bash
# Automatic device selection
python scripts/inference.py --model-path checkpoints/best_model

# Force CUDA
python scripts/inference.py --model-path checkpoints/best_model --device cuda
```

## Troubleshooting

### CUDA Not Detected

1. Check CUDA installation:
   ```bash
   nvidia-smi
   ```

2. Verify PyTorch CUDA support:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.version.cuda)
   ```

3. Install CUDA-enabled PyTorch:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Out of Memory Errors

1. Reduce batch size:
   ```bash
   python scripts/train_model.py --batch-size 4
   ```

2. Disable mixed precision:
   ```python
   config.training.use_mixed_precision = False
   ```

3. Use gradient accumulation (in config):
   ```python
   config.training.gradient_accumulation_steps = 4
   ```

### Performance Issues

1. Check GPU utilization:
   ```bash
   nvidia-smi -l 1
   ```

2. Ensure data is on GPU:
   - The trainer automatically handles device placement
   - Check logs for device transfer warnings

3. Profile training:
   ```bash
   python scripts/train_model.py --profile
   ```

## Multi-GPU Support

For multi-GPU training (future enhancement):

```python
# DataParallel (simple but less efficient)
model = nn.DataParallel(model)

# DistributedDataParallel (recommended)
# Implementation coming soon
```

## Edge Deployment

While training benefits from CUDA, the model is optimized for edge deployment on NVIDIA Jetson Nano:

- Model quantization for INT8 inference
- TensorRT optimization (optional)
- Batch inference support
- Memory-efficient inference pipeline

## Environment Variables

Useful environment variables for CUDA:

```bash
# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Enable TF32 on Ampere GPUs
export TORCH_ALLOW_TF32=1

# Disable CUDA for debugging
export CUDA_VISIBLE_DEVICES=""
```

## Best Practices

1. **Always check device availability** before training
2. **Monitor GPU memory** during training
3. **Use mixed precision** for faster training (CUDA only)
4. **Clear cache** between experiments
5. **Profile your code** to identify bottlenecks

## Performance Monitoring

During training, the following metrics are logged:

- Device type and memory usage
- Training/validation time per epoch
- Inference time per batch
- Memory allocation statistics

Check `training.log` for detailed device performance metrics. 