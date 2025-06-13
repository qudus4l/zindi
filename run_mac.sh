#!/bin/bash
# Run script for Mac with Apple Silicon (MPS)

echo "=========================================="
echo "Clinical Decision Support Model Training"
echo "=========================================="

# Run training with MPS
python3 scripts/train_model.py \
    --device mps \
    --epochs 5 \
    --batch-size 8

echo -e "\n\nTraining complete!" 