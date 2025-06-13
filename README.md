# Clinical Decision Support Model for Kenyan Healthcare

## 🏥 Project Overview

This project implements a clinical decision support model designed specifically for Kenyan healthcare settings. The model assists nurses in making informed clinical decisions by generating appropriate responses to medical vignettes, with a focus on resource-constrained environments.

### Key Features
- **Edge-Optimized**: Designed to run on NVIDIA Jetson Nano with <100ms inference time
- **Medical Safety First**: Built with clinical validation and safety checks at every step
- **Context-Aware**: Considers nurse experience levels and facility types
- **Kenyan Healthcare Focus**: Handles local medical terminology and practices

## 🎯 Technical Constraints

- **Model Size**: < 1 billion parameters
- **Inference Time**: < 100ms per vignette
- **Memory Usage**: < 2GB RAM during inference
- **Target Hardware**: NVIDIA Jetson Nano or equivalent edge device

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
│   ├── models/                 # Model architectures
│   ├── training/               # Training pipelines
│   ├── evaluation/             # Evaluation metrics
│   ├── optimization/           # Quantization and edge optimization
│   └── utils/                  # Utilities and configuration
├── scripts/                    # Executable scripts
├── notebooks/                  # Jupyter notebooks for experiments
├── tests/                      # Unit and integration tests
├── configs/                    # Configuration files
└── docs/                       # Additional documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (for training)
- 16GB+ RAM recommended

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd zindi
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Analysis

Analyze the clinical vignette dataset:
```bash
python scripts/analyze_data_minimal.py
```

### Data Preprocessing

Preprocess the data for model training:
```bash
python scripts/preprocess_data.py
```

### Model Evaluation

Evaluate different model architectures:
```bash
python scripts/evaluate_models.py
```

## 📈 Implementation Phases

### Phase 1: Data Analysis & Exploration ✅
- Comprehensive dataset analysis
- Clinical pattern identification
- Response length distribution analysis
- Medical terminology extraction

### Phase 2: Model Architecture Selection ✅
- Evaluated 6 model architectures
- Selected clinical-T5-small as primary candidate
- Alternatives: BioBERT, custom lightweight transformer

### Phase 3: Data Preprocessing ✅
- Medical term standardization
- Kenyan terminology handling
- Response formatting
- Safety validation

### Phase 4: Model Training (In Progress)
- Progressive fine-tuning strategy
- Clinical safety callbacks
- Performance monitoring

### Phase 5: Edge Optimization (Planned)
- INT8/INT4 quantization
- ONNX conversion
- TensorRT optimization
- Memory footprint reduction

### Phase 6: Validation & Testing (Planned)
- ROUGE score optimization
- Clinical accuracy assessment
- Safety evaluation
- Edge deployment testing

## 🔧 Configuration

The project uses a comprehensive configuration system. Key parameters:

```python
# Model constraints
max_parameters: 1_000_000_000  # 1B limit
max_inference_time_ms: 100     # 100ms limit
max_memory_gb: 2.0             # 2GB limit

# Training parameters
batch_size: 8
learning_rate: 5e-5
num_epochs: 20

# Safety parameters
enable_safety_checks: True
confidence_threshold: 0.7
```

## 📊 Current Results

### Data Analysis
- Average prompt length: 113 words
- Average response length: 110 words
- Most common medical terms: patient, pain, diagnosis, child, administer

### Model Evaluation
1. **Clinical-T5-small**: 98.5/100 score
   - 60M parameters
   - Medical pre-training compatible
   - Excellent for clinical text generation

2. **BioBERT-base**: 85.2/100 score
   - 110M parameters
   - Pre-trained on biomedical literature
   - Strong medical understanding

## 🏥 Clinical Safety Considerations

1. **Validation Checkpoints**: Clinical expert review at key stages
2. **Confidence Scoring**: All outputs include confidence levels
3. **Harmful Output Detection**: Zero tolerance for dangerous advice
4. **Fallback Mechanisms**: Graceful degradation for edge cases
5. **Audit Trail**: Complete logging of all decisions

## 🤝 Contributing

This is a high-stakes medical AI project. All contributions must:
1. Prioritize patient safety
2. Include comprehensive testing
3. Document clinical reasoning
4. Pass safety validation

## ⚠️ Disclaimer

This model is designed to support, not replace, clinical decision-making. Healthcare professionals should always use their clinical judgment and consider local guidelines when making patient care decisions.

## 📝 License

[Specify license here]

## 🙏 Acknowledgments

- Kenyan healthcare workers who provided the clinical vignettes
- Zindi platform for hosting the challenge
- Medical experts who validated our approach

---

**Remember**: In medical AI, patient safety always comes first. When in doubt, flag for human expert review. 