#!/bin/bash
# Setup and run script for Shadeform GPU instances

echo "Setting up Clinical Decision Support Model Training..."

# Ensure we're in the right directory
cd /home/shadeform/zindi

# Install dependencies if not already installed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install torch transformers pandas numpy scikit-learn rouge-score sentencepiece tqdm

# Add project root to PYTHONPATH
export PYTHONPATH=/home/shadeform/zindi:$PYTHONPATH

# Check CUDA availability
echo -e "\n\nChecking CUDA..."
python scripts/check_device.py

# Run training - use python from venv, not python3
echo -e "\n\nStarting training with CUDA..."
cd /home/shadeform/zindi && python scripts/train_model.py --device cuda --epochs 5 --batch-size 32

echo -e "\n\nTraining complete!" 