# Quick Start - Run Everything

## One Command to Run Everything

### Option 1: Bash Script (Recommended)
```bash
./run_training.sh
```

### Option 2: Python Script
```bash
python run_all.py
```

### Option 3: Direct Command
```bash
python scripts/train_model.py --device cuda --epochs 5 --batch-size 16
```

## What It Does

1. **Checks CUDA availability** - Verifies your GPU is detected
2. **Trains the model** - 5 epochs with batch size 16 (takes ~10-15 minutes on a good GPU)
3. **Saves results** - Model checkpoints and training reports

## Settings Used

- **Epochs**: 5 (reasonable for initial training)
- **Batch Size**: 16 (good for most GPUs with 8GB+ VRAM)
- **Device**: CUDA (automatically uses your GPU)
- **Model**: T5-small (60M parameters, fits edge constraints)

## After Training

Generate predictions on test data:
```bash
python scripts/inference.py --model-path checkpoints/epoch_4/ --device cuda
```

## Expected Training Time

- **RTX 3090/4090**: ~10 minutes
- **RTX 3080**: ~12 minutes  
- **RTX 3070**: ~15 minutes
- **Older GPUs**: 20-30 minutes

## If Something Goes Wrong

1. Check CUDA:
   ```bash
   python scripts/check_device.py
   ```

2. Reduce batch size if out of memory:
   ```bash
   python scripts/train_model.py --device cuda --epochs 5 --batch-size 8
   ```

3. Check logs:
   ```bash
   tail -f training.log
   ``` 