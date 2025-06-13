"""Configuration management for the clinical decision support system.

This module handles all configuration parameters with a focus on clinical
safety and edge device constraints.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data pipeline configuration.
    
    Attributes:
        raw_data_dir: Directory containing raw clinical data
        processed_data_dir: Directory for preprocessed data
        augmented_data_dir: Directory for augmented training data
        train_file: Training data filename
        test_file: Test data filename
        val_split: Validation split ratio
        max_sequence_length: Maximum input sequence length
        min_response_length: Minimum valid response length
        max_response_length: Maximum response length
    """
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    augmented_data_dir: Path = Path("data/augmented")
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    val_split: float = 0.2
    max_sequence_length: int = 512
    min_response_length: int = 10
    max_response_length: int = 200


@dataclass
class ModelConfig:
    """Model architecture configuration.
    
    Attributes:
        model_type: Type of model architecture
        max_parameters: Maximum allowed parameters (1B limit)
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout_rate: Dropout probability
        vocab_size: Vocabulary size
        medical_vocab_size: Additional medical vocabulary
    """
    model_type: str = "t5-small"
    max_parameters: int = 1_000_000_000  # 1B parameter limit
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout_rate: float = 0.1
    vocab_size: int = 32000
    medical_vocab_size: int = 5000


@dataclass
class TrainingConfig:
    """Training configuration with clinical safety parameters.
    
    Attributes:
        batch_size: Training batch size
        learning_rate: Initial learning rate
        num_epochs: Maximum training epochs
        warmup_steps: Learning rate warmup steps
        gradient_clip_norm: Gradient clipping threshold
        early_stopping_patience: Early stopping patience
        clinical_validation_frequency: Frequency of clinical validation
        safety_check_frequency: Frequency of safety checks
        use_mixed_precision: Enable mixed precision training
        device: Preferred device ('cuda', 'mps', 'cpu', or 'auto')
    """
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 20
    warmup_steps: int = 500
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 3
    clinical_validation_frequency: int = 100
    safety_check_frequency: int = 50
    use_mixed_precision: bool = True
    device: str = 'auto'  # 'cuda', 'mps', 'cpu', or 'auto' for automatic selection


@dataclass
class OptimizationConfig:
    """Edge device optimization configuration.
    
    Attributes:
        target_device: Target deployment device
        max_inference_time_ms: Maximum inference time in milliseconds
        max_memory_gb: Maximum RAM usage in GB
        quantization_type: Type of quantization (int8, int4)
        use_onnx: Convert to ONNX format
        use_tensorrt: Use TensorRT optimization
        batch_inference: Enable batch inference
    """
    target_device: str = "jetson_nano"
    max_inference_time_ms: int = 100  # 100ms limit
    max_memory_gb: float = 2.0  # 2GB RAM limit
    quantization_type: str = "int8"
    use_onnx: bool = True
    use_tensorrt: bool = False
    batch_inference: bool = True


@dataclass
class ClinicalSafetyConfig:
    """Clinical safety and validation configuration.
    
    Attributes:
        enable_safety_checks: Enable all safety validations
        harmful_output_threshold: Threshold for harmful content detection
        confidence_threshold: Minimum confidence for predictions
        require_reasoning_chain: Require explicit reasoning
        flag_uncertain_cases: Flag low-confidence predictions
        medical_term_validation: Validate medical terminology
        cultural_sensitivity_check: Check cultural appropriateness
    """
    enable_safety_checks: bool = True
    harmful_output_threshold: float = 0.95
    confidence_threshold: float = 0.7
    require_reasoning_chain: bool = True
    flag_uncertain_cases: bool = True
    medical_term_validation: bool = True
    cultural_sensitivity_check: bool = True


@dataclass
class Config:
    """Main configuration class combining all sub-configurations.
    
    Attributes:
        data: Data pipeline configuration
        model: Model architecture configuration
        training: Training configuration
        optimization: Edge optimization configuration
        clinical_safety: Clinical safety configuration
        project_root: Project root directory
        experiment_name: Current experiment name
        seed: Random seed for reproducibility
    """
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    clinical_safety: ClinicalSafetyConfig = field(default_factory=ClinicalSafetyConfig)
    project_root: Path = Path(os.getcwd())
    experiment_name: str = "clinical_decision_support"
    seed: int = 42

    def save(self, filepath: Path) -> None:
        """Save configuration to YAML file.
        
        Args:
            filepath: Path to save configuration file
        """
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'optimization': self.optimization.__dict__,
            'clinical_safety': self.clinical_safety.__dict__,
            'experiment_name': self.experiment_name,
            'seed': self.seed
        }
        
        # Convert Path objects to strings
        for section in config_dict.values():
            if isinstance(section, dict):
                for key, value in section.items():
                    if isinstance(value, Path):
                        section[key] = str(value)
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        logger.info(f"Configuration saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'Config':
        """Load configuration from YAML file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            Config: Loaded configuration object
        """
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert string paths back to Path objects
        if 'data' in config_dict:
            for key in ['raw_data_dir', 'processed_data_dir', 'augmented_data_dir']:
                if key in config_dict['data']:
                    config_dict['data'][key] = Path(config_dict['data'][key])
        
        # Create sub-configurations
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        optimization_config = OptimizationConfig(**config_dict.get('optimization', {}))
        clinical_safety_config = ClinicalSafetyConfig(**config_dict.get('clinical_safety', {}))
        
        config = cls(
            data=data_config,
            model=model_config,
            training=training_config,
            optimization=optimization_config,
            clinical_safety=clinical_safety_config,
            experiment_name=config_dict.get('experiment_name', 'clinical_decision_support'),
            seed=config_dict.get('seed', 42)
        )
        
        logger.info(f"Configuration loaded from {filepath}")
        return config

    def validate(self) -> List[str]:
        """Validate configuration for clinical safety and technical constraints.
        
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        
        # Technical constraints validation
        if self.optimization.max_inference_time_ms > 100:
            errors.append(f"Inference time {self.optimization.max_inference_time_ms}ms exceeds 100ms limit")
        
        if self.optimization.max_memory_gb > 2.0:
            errors.append(f"Memory usage {self.optimization.max_memory_gb}GB exceeds 2GB limit")
        
        if self.model.max_parameters > 1_000_000_000:
            errors.append(f"Model parameters exceed 1B limit")
        
        # Clinical safety validation
        if not self.clinical_safety.enable_safety_checks:
            errors.append("Clinical safety checks must be enabled for medical AI")
        
        if self.clinical_safety.confidence_threshold < 0.5:
            errors.append("Confidence threshold too low for medical decisions")
        
        # Data validation
        if self.data.val_split <= 0 or self.data.val_split >= 1:
            errors.append("Validation split must be between 0 and 1")
        
        return errors


# Create default configuration instance
default_config = Config()

# Validate default configuration
validation_errors = default_config.validate()
if validation_errors:
    logger.warning(f"Configuration validation errors: {validation_errors}") 