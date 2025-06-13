#!/usr/bin/env python3
"""Phase 4: Model Training Script.

This script trains the clinical decision support model with safety
monitoring and performance optimization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import logging
from pathlib import Path
import json
import argparse
from datetime import datetime

from src.models.clinical_t5 import ClinicalT5Model, ClinicalT5Config
from src.training.trainer import ClinicalTrainer, ClinicalDataset
from src.utils.config import Config
from src.utils.device_utils import get_device_info, get_optimal_device
from src.evaluation.metrics import ClinicalMetricsEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_model(config: Config) -> ClinicalT5Model:
    """Setup the clinical T5 model.
    
    Args:
        config: Configuration object
        
    Returns:
        ClinicalT5Model: Initialized model
    """
    logger.info("Setting up Clinical T5 model")
    
    # Create model configuration
    model_config = ClinicalT5Config(
        model_name=config.model.model_type,
        max_input_length=config.data.max_sequence_length,
        max_output_length=config.data.max_response_length,
        confidence_threshold=config.clinical_safety.confidence_threshold
    )
    
    # Initialize model
    model = ClinicalT5Model(model_config)
    
    # Setup base T5 model
    model.setup_model()
    
    # Log model info
    num_params = model.get_num_parameters()
    logger.info(f"Model initialized with {num_params:,} parameters")
    
    if num_params > config.model.max_parameters:
        logger.warning(f"Model exceeds parameter limit: {num_params:,} > {config.model.max_parameters:,}")
    
    return model


def prepare_datasets(config: Config, tokenizer) -> tuple:
    """Prepare training and validation datasets.
    
    Args:
        config: Configuration object
        tokenizer: Model tokenizer
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    logger.info("Preparing datasets")
    
    # Create datasets
    train_dataset = ClinicalDataset(
        data_path=config.data.processed_data_dir / "train_processed.json",
        tokenizer=tokenizer,
        max_length=config.data.max_sequence_length
    )
    
    val_dataset = ClinicalDataset(
        data_path=config.data.processed_data_dir / "val_processed.json",
        tokenizer=tokenizer,
        max_length=config.data.max_sequence_length
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Validation dataset: {len(val_dataset)} samples")
    
    return train_dataset, val_dataset


def train_model(config: Config, model: ClinicalT5Model, 
                train_dataset: ClinicalDataset, 
                val_dataset: ClinicalDataset) -> dict:
    """Train the model.
    
    Args:
        config: Configuration object
        model: Model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        
    Returns:
        dict: Training results
    """
    logger.info("Starting model training")
    
    # Log device information
    device_info = get_device_info()
    logger.info(f"Available devices: {device_info}")
    
    # Create trainer (device selection handled internally)
    trainer = ClinicalTrainer(
        model=model,
        config=config
    )
    
    # Train model
    results = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=config.training.num_epochs
    )
    
    return results


def evaluate_final_model(model_path: str, val_dataset: ClinicalDataset, config: Config):
    """Evaluate the final trained model.
    
    Args:
        model_path: Path to best model checkpoint
        val_dataset: Validation dataset
        config: Configuration object
    """
    logger.info("Evaluating final model")
    
    # Load model
    model = ClinicalT5Model(ClinicalT5Config())
    model.load_pretrained(model_path)
    model.eval()
    
    # Get optimal device
    device = get_optimal_device(config.training.device)
    model.to(device)
    logger.info(f"Evaluating on device: {device}")
    
    # Generate predictions
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    all_predictions = []
    all_references = []
    inference_times = []
    
    import time
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Measure inference time
            start_time = time.time()
            
            generated_ids = model.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=config.data.max_response_length,
                num_beams=4,
                early_stopping=True
            )
            
            inference_time = (time.time() - start_time) / len(batch['input_ids'])
            inference_times.append(inference_time * 1000)  # Convert to ms
            
            # Decode
            predictions = model.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            references = model.tokenizer.batch_decode(
                batch['labels'], skip_special_tokens=True
            )
            
            all_predictions.extend(predictions)
            all_references.extend(references)
    
    # Calculate metrics
    evaluator = ClinicalMetricsEvaluator()
    results = evaluator.evaluate_batch(all_predictions, all_references)
    
    # Add inference time stats
    results['inference_stats'] = {
        'avg_inference_time_ms': np.mean(inference_times),
        'max_inference_time_ms': np.max(inference_times),
        'min_inference_time_ms': np.min(inference_times),
        'meets_constraint': np.mean(inference_times) < config.optimization.max_inference_time_ms
    }
    
    # Print results
    print("\n" + "=" * 70)
    print("FINAL MODEL EVALUATION")
    print("=" * 70)
    print(evaluator.format_results(results))
    print(f"\nInference Time: {results['inference_stats']['avg_inference_time_ms']:.2f}ms (avg)")
    print(f"Meets <100ms constraint: {results['inference_stats']['meets_constraint']}")
    print("=" * 70)
    
    return results


def save_training_report(results: dict, config: Config):
    """Save comprehensive training report.
    
    Args:
        results: Training results
        config: Configuration object
    """
    report_dir = Path("training_results")
    report_dir.mkdir(exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results
    report_path = report_dir / f"training_report_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create summary report
    summary_path = report_dir / f"training_summary_{timestamp}.txt"
    with open(summary_path, 'w') as f:
        f.write("CLINICAL DECISION SUPPORT MODEL TRAINING SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Training Configuration:\n")
        f.write(f"  Model: {config.model.model_type}\n")
        f.write(f"  Epochs: {config.training.num_epochs}\n")
        f.write(f"  Batch Size: {config.training.batch_size}\n")
        f.write(f"  Learning Rate: {config.training.learning_rate}\n\n")
        
        f.write("Results:\n")
        f.write(f"  Best ROUGE-1 F1: {results['best_rouge_score']:.4f}\n")
        f.write(f"  Best Model Path: {results['best_model_path']}\n")
        f.write(f"  Total Safety Violations: {results['safety_report']['total_violations']}\n\n")
        
        f.write("Safety Report:\n")
        for violation_type, count in results['safety_report']['violation_types'].items():
            f.write(f"  {violation_type}: {count}\n")
    
    logger.info(f"Training report saved to {report_path}")
    logger.info(f"Training summary saved to {summary_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train clinical decision support model")
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--device', type=str, choices=['cuda', 'mps', 'cpu', 'auto'], 
                        default='auto', help='Device to use for training')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation')
    parser.add_argument('--model-path', type=str, help='Path to model for evaluation')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = Config.load(Path(args.config))
    else:
        config = Config()
    
    # Override config with command line args
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.device:
        config.training.device = args.device
    
    # Validate configuration
    validation_errors = config.validate()
    if validation_errors:
        logger.error(f"Configuration validation failed: {validation_errors}")
        return
    
    try:
        if args.eval_only:
            # Evaluation only mode
            if not args.model_path:
                logger.error("--model-path required for evaluation mode")
                return
            
            # Setup model for tokenizer
            model = setup_model(config)
            _, val_dataset = prepare_datasets(config, model.tokenizer)
            
            # Evaluate
            evaluate_final_model(args.model_path, val_dataset, config)
        else:
            # Full training mode
            # Setup model
            model = setup_model(config)
            
            # Prepare datasets
            train_dataset, val_dataset = prepare_datasets(config, model.tokenizer)
            
            # Train model
            results = train_model(config, model, train_dataset, val_dataset)
            
            # Evaluate final model
            if results['best_model_path']:
                eval_results = evaluate_final_model(
                    results['best_model_path'], 
                    val_dataset, 
                    config
                )
                results['final_evaluation'] = eval_results
            
            # Save report
            save_training_report(results, config)
            
            print("\n" + "=" * 70)
            print("TRAINING COMPLETE")
            print("=" * 70)
            print(f"Best model saved to: {results['best_model_path']}")
            print(f"Best ROUGE-1 F1 score: {results['best_rouge_score']:.4f}")
            print("=" * 70)
            
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import numpy as np  # Import numpy for evaluation
    main() 