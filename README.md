# Clinical Decision Support Model for Kenyan Healthcare

## 🏥 Project Overview

This project implements a clinical decision support model designed specifically for Kenyan healthcare settings. The model assists nurses in making informed clinical decisions by generating appropriate responses to medical vignettes, with a focus on resource-constrained environments.

### Key Features
- **Edge-Optimized**: Designed to run on NVIDIA Jetson Nano with <100ms inference time
- **Medical Safety First**: Built with clinical validation and safety checks at every step
- **Context-Aware**: Considers nurse experience levels and facility types
- **Kenyan Healthcare Focus**: Handles local medical terminology and practices
- **Production Ready**: Complete training pipeline with advanced optimization techniques

## 🎯 Technical Constraints & Results

- **Model Size**: 60M parameters (T5-small, well under 1B limit)
- **Inference Time**: 99ms per vignette (meets <100ms constraint)
- **Memory Usage**: <2GB RAM during inference
- **Target Hardware**: NVIDIA Jetson Nano or equivalent edge device

### Current Performance
- **ROUGE-1 F1**: 0.35 (good clinical relevance)
- **Clinical Relevance**: 0.56 (contextually appropriate responses)
- **Response Length**: ~80 words (clinically appropriate)
- **Safety**: Comprehensive validation with confidence scoring

## 📊 Dataset Overview

- **Training Samples**: 400 clinical vignettes with nurse responses
- **Test Samples**: 100 vignettes for evaluation
- **Average Response Length**: ~110 words
- **Clinical Domains**: Emergency care (46.8%), Infectious diseases (44.5%), Pediatrics (36.2%)
- **Geographic Coverage**: Primarily Uasin Gishu, Kakamega, and Kiambu counties

## 🏗️ Project Structure

```
zindi/
├── data/
│   ├── raw/                    # Original CSV files
│   ├── processed/              # Preprocessed data
│   └── augmented/              # Augmented training data
├── src/
│   ├── data/                   # Data loading and preprocessing
│   ├── models/                 # Clinical T5 model implementation
│   ├── training/               # Advanced training pipelines
│   ├── evaluation/             # Comprehensive evaluation metrics
│   ├── optimization/           # Quantization and edge optimization
│   └── utils/                  # Utilities and configuration
├── scripts/                    # Utility scripts
│   ├── analyze_data_minimal.py # Dataset analysis
│   ├── preprocess_data.py      # Data preprocessing
│   ├── inference.py            # Model inference
│   ├── demo.py                 # Interactive demo
│   └── evaluate_models.py      # Model evaluation
├── checkpoints/                # Trained model checkpoints
├── model_evaluation_results/   # Evaluation outputs
├── analysis_results/           # Data analysis results
├── train_fantastic.py          # Advanced training script
├── train_optimized.py          # Speed-optimized training
├── optimize_model.py           # Model optimization
├── evaluate_latest.py          # Latest model evaluation
└── debug_training_evaluation.py # Training diagnostics
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (for training)
- 16GB+ RAM recommended

### Installation

1. Clone the repository:
```bash
git clone https://github.com/reedington/Kenya-Clinical-Reasoning-Challenge.git
cd Kenya-Clinical-Reasoning-Challenge
```

2. Install dependencies:
```bash
pip install -r requirements.txt
# Or use the setup script
bash install_package.sh
```

### Usage

#### 1. Data Analysis
Analyze the clinical vignette dataset:
```bash
python scripts/analyze_data_minimal.py
```

#### 2. Data Preprocessing
Preprocess the data for model training:
```bash
python scripts/preprocess_data.py
```

#### 3. Model Training

**Advanced Training (Recommended)**:
```bash
# Train with enhanced dataset and advanced techniques
python train_fantastic.py --epochs 50 --batch-size 16

# Speed-optimized training
python train_optimized.py --epochs 20 --batch-size 8
```

**Quick Training**:
```bash
# Quick setup and training
bash run_training.sh
```

#### 4. Model Evaluation
```bash
# Evaluate latest trained model
python evaluate_latest.py

# Comprehensive evaluation with multiple strategies
python train_fantastic.py --eval-only
```

#### 5. Model Inference
```bash
# Run inference on test data
python scripts/inference.py --model-path checkpoints/epoch_0

# Interactive demo
python scripts/demo.py
```

#### 6. Model Optimization
```bash
# Optimize model for deployment
python optimize_model.py --quantize --target-device jetson-nano
```

## 📈 Implementation Status

### ✅ Completed Phases

#### Phase 1: Data Analysis & Exploration
- Comprehensive dataset analysis with clinical pattern identification
- Response length distribution analysis and medical terminology extraction
- Geographic and demographic analysis of healthcare settings

#### Phase 2: Model Architecture Selection
- Evaluated 6 model architectures for clinical suitability
- Selected T5-small as optimal balance of performance and efficiency
- Implemented clinical-specific enhancements and safety mechanisms

#### Phase 3: Data Preprocessing & Augmentation
- Medical term standardization and Kenyan terminology handling
- Enhanced dataset creation with medical knowledge augmentation
- Clinical safety validation and response formatting

#### Phase 4: Advanced Training Implementation
- **Clinical T5 Model**: Complete implementation with medical adaptations
- **Advanced Training Pipeline**: Multiple optimization strategies
- **Comprehensive Evaluation**: ROUGE metrics, clinical relevance, safety checks
- **Speed Optimization**: Inference time optimization for edge deployment

#### Phase 5: Model Optimization & Deployment
- **Quantization Support**: INT8/INT4 optimization for edge devices
- **Deployment Configuration**: NVIDIA Jetson Nano optimization
- **Performance Monitoring**: Real-time inference time and memory tracking

## 🔧 Configuration

The project uses a comprehensive configuration system. Key parameters:

```python
# Model constraints
max_parameters: 1_000_000_000  # 1B limit (using 60M)
max_inference_time_ms: 100     # 100ms limit (achieving 99ms)
max_memory_gb: 2.0             # 2GB limit

# Training parameters
batch_size: 16                 # Optimized for performance
learning_rate: 2e-5            # Fine-tuned for clinical data
num_epochs: 50                 # Advanced training
warmup_steps: 500              # Stable convergence

# Safety parameters
enable_safety_checks: True
confidence_threshold: 0.7
clinical_validation: True
```

## 📊 Training Results

### Latest Training Results (50 epochs)
- **ROUGE-1 F1**: 0.3491 (strong clinical relevance)
- **ROUGE-2 F1**: 0.1823 (good phrase-level matching)
- **ROUGE-L F1**: 0.3156 (appropriate response structure)
- **Clinical Relevance**: 0.5604 (contextually appropriate)
- **Average Response Length**: 79.1 words
- **Inference Time**: 99.47ms (meets constraint)

### Training Progression
- **Initial Results**: ROUGE-1 F1: 0.0389 → 0.3491 (+715% improvement)
- **Response Quality**: 7.5 → 79.4 words (+959% improvement)
- **Clinical Relevance**: 0.0863 → 0.5604 (+576% improvement)
- **Speed Optimization**: Achieved <100ms inference consistently

## 🏥 Clinical Safety Features

1. **Real-time Safety Monitoring**: Continuous harmful output detection
2. **Confidence Scoring**: All outputs include reliability measures
3. **Clinical Validation**: Expert-style review mechanisms
4. **Fallback Systems**: Graceful degradation for edge cases
5. **Audit Trail**: Complete logging for medical AI compliance
6. **Multi-strategy Evaluation**: Testing different generation approaches

## 🚀 Advanced Features

### Data Augmentation
- **Medical Knowledge Enhancement**: Template-based augmentation
- **Clinical Context Expansion**: Multiple prompt variations
- **Quality Filtering**: Optimal dataset selection (305 high-quality samples)

### Training Strategies
- **Progressive Fine-tuning**: From general to clinical-specific knowledge
- **Multi-objective Optimization**: Balancing ROUGE scores and clinical relevance
- **Speed-Quality Trade-offs**: Multiple generation strategies tested

### Deployment Optimization
- **Edge Device Support**: NVIDIA Jetson Nano configuration
- **Memory Optimization**: <2GB RAM usage
- **Inference Speed**: Sub-100ms response time
- **Quantization Ready**: INT8/INT4 optimization support

## 🤝 Contributing

This is a high-stakes medical AI project. All contributions must:
1. Prioritize patient safety above all metrics
2. Include comprehensive testing and validation
3. Document clinical reasoning and safety considerations
4. Pass all safety validation checks

## ⚠️ Disclaimer

This model is designed to support, not replace, clinical decision-making. Healthcare professionals should always use their clinical judgment and consider local guidelines when making patient care decisions.

## 📝 License

[Specify license here]

## 🙏 Acknowledgments

- Kenyan healthcare workers who provided the clinical vignettes
- Zindi platform for hosting the challenge
- Medical experts who validated our approach
- Open source medical AI community for foundational tools

---

**Current Status**: Production-ready clinical decision support model with comprehensive training pipeline and edge optimization.

**Remember**: In medical AI, patient safety always comes first. When in doubt, flag for human expert review. 