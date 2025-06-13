#!/bin/bash
# One-command script to run the entire clinical decision support model training

echo "=========================================="
echo "Clinical Decision Support Model Training"
echo "=========================================="

# Check if CUDA is available
echo "Checking CUDA availability..."
python scripts/check_device.py

echo -e "\n\nStarting training pipeline..."

# Run training with reasonable settings for quick results
# 5 epochs should give decent results while not taking too long
python scripts/train_model.py \
    --device cuda \
    --epochs 5 \
    --batch-size 16

echo -e "\n\nTraining complete! Check the results in:"
echo "- training_results/ (for training reports)"
echo "- checkpoints/ (for saved models)"
echo "- training.log (for detailed logs)" 