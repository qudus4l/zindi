#!/bin/bash
# Installation script for Zindi Clinical AI package

set -e  # Exit on any error

echo "🚀 Installing Zindi Clinical AI package in development mode..."

# Check if we're in the right directory
if [ ! -f "setup.py" ] || [ ! -d "src" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   Make sure you can see setup.py and src/ directory"
    exit 1
fi

# Install the package in development mode
echo "📦 Installing package in editable/development mode..."
pip install -e .

# Verify installation
echo "🔍 Verifying installation..."
python3 -c "
import sys
try:
    from src.models.clinical_t5 import ClinicalT5Model, ClinicalT5Config
    from src.training.trainer import ClinicalTrainer, ClinicalDataset
    from src.utils.config import Config
    print('✅ All imports successful!')
    print('✅ Package installed correctly!')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

echo ""
echo "🎉 Installation complete!"
echo ""
echo "You can now run:"
echo "  python3 scripts/train_model.py --device cuda --epochs 5 --batch-size 32"
echo ""
echo "Or use the installed console commands:"
echo "  train-clinical-model --device cuda --epochs 5 --batch-size 32"
echo "" 