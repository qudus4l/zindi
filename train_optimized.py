#!/usr/bin/env python3
"""
Optimized training script for fantastic clinical model performance.
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import argparse
import torch
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.models.clinical_t5 import ClinicalT5Model, ClinicalT5Config
from src.training.trainer import ClinicalDataset, ClinicalTrainer
from src.evaluation.metrics import ClinicalMetricsEvaluator
from src.utils.config import Config

def setup_optimized_model():
    """Setup model with optimized configuration."""
    logger.info("Setting up optimized Clinical T5 model")
    
    # Use optimized config
    model_config = ClinicalT5Config(
        model_name="t5-small",
        max_input_length=512,
        max_output_length=150,  # Reduced for speed
        num_beams=2,           # Faster generation
        temperature=0.7,       # Slightly more focused
        repetition_penalty=1.3, # Avoid repetition
        length_penalty=1.1     # Encourage longer responses
    )
    
    model = ClinicalT5Model(model_config)
    model.setup_model()
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.model.parameters()):,} parameters")
    return model

def prepare_optimized_datasets(model):
    """Prepare datasets using optimal training data."""
    logger.info("Preparing optimized datasets")
    
    # Use optimal training data
    train_dataset = ClinicalDataset(
        data_path="data/processed/train_optimal.json",  # Use filtered data
        tokenizer=model.tokenizer,
        max_length=512
    )
    
    val_dataset = ClinicalDataset(
        data_path="data/processed/val_processed.json",
        tokenizer=model.tokenizer,
        max_length=512
    )
    
    logger.info(f"Optimal train dataset: {len(train_dataset)} samples")
    logger.info(f"Validation dataset: {len(val_dataset)} samples")
    
    return train_dataset, val_dataset

def train_optimized_model(epochs=30, batch_size=4):
    """Train model with optimized parameters."""
    logger.info("Starting optimized training")
    
    # Setup model
    model = setup_optimized_model()
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_optimized_datasets(model)
    
    # Create optimized config
    config = Config()
    
    # Optimize training parameters
    config.training.num_epochs = epochs
    config.training.batch_size = batch_size
    config.training.learning_rate = 3e-5  # Slightly lower for stability
    config.training.warmup_steps = 100    # Reduced warmup
    config.training.early_stopping_patience = 5  # More patience
    config.training.gradient_clip_norm = 0.5     # Tighter clipping
    
    # Setup trainer
    trainer = ClinicalTrainer(model, config)
    trainer.setup_optimization()
    
    # Train
    results = trainer.train(train_dataset, val_dataset, num_epochs=epochs)
    
    logger.info("Optimized training complete")
    return results, model, val_dataset

def evaluate_optimized_model(model, val_dataset):
    """Evaluate the optimized model."""
    logger.info("Evaluating optimized model")
    
    model.eval()
    evaluator = ClinicalMetricsEvaluator()
    
    # Test on validation set
    predictions = []
    references = []
    inference_times = []
    
    import time
    from torch.utils.data import DataLoader
    
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    with torch.no_grad():
        for batch in val_loader:
            # Generate predictions
            start_time = time.time()
            
            generated_ids = model.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_new_tokens=150,
                min_length=30,
                num_beams=2,
                early_stopping=True,
                do_sample=False,
                repetition_penalty=1.3,
                length_penalty=1.1
            )
            
            inference_time = (time.time() - start_time) / len(batch['input_ids'])
            inference_times.append(inference_time * 1000)
            
            # Decode predictions
            batch_predictions = model.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            predictions.extend(batch_predictions)
            
            # Decode references
            labels_for_decode = batch['labels'].clone()
            labels_for_decode[labels_for_decode == -100] = model.tokenizer.pad_token_id
            batch_references = model.tokenizer.batch_decode(
                labels_for_decode, skip_special_tokens=True
            )
            references.extend(batch_references)
    
    # Evaluate
    results = evaluator.evaluate_batch(predictions, references)
    
    # Add inference stats
    results['inference_stats'] = {
        'avg_inference_time_ms': sum(inference_times) / len(inference_times),
        'max_inference_time_ms': max(inference_times),
        'meets_constraint': (sum(inference_times) / len(inference_times)) < 100
    }
    
    # Print results
    print("\n" + "=" * 70)
    print("OPTIMIZED MODEL EVALUATION")
    print("=" * 70)
    print(evaluator.format_results(results))
    print(f"\nInference Time: {results['inference_stats']['avg_inference_time_ms']:.2f}ms (avg)")
    print(f"Meets <100ms constraint: {results['inference_stats']['meets_constraint']}")
    print("=" * 70)
    
    return results

def main():
    """Main optimized training function."""
    parser = argparse.ArgumentParser(description="Train optimized clinical model")
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate existing model')
    
    args = parser.parse_args()
    
    if args.eval_only:
        # Load existing model and evaluate
        model = setup_optimized_model()
        if Path("checkpoints/epoch_0").exists():
            model.load_pretrained("checkpoints/epoch_0")
            _, val_dataset = prepare_optimized_datasets(model)
            evaluate_optimized_model(model, val_dataset)
        else:
            logger.error("No checkpoint found for evaluation")
    else:
        # Full training
        results, model, val_dataset = train_optimized_model(args.epochs, args.batch_size)
        
        # Final evaluation
        final_results = evaluate_optimized_model(model, val_dataset)
        
        print(f"\n🎯 OPTIMIZATION COMPLETE!")
        print(f"Best ROUGE-1 F1: {final_results['summary']['rouge1_f1']:.4f}")
        print(f"Clinical Relevance: {final_results['summary']['clinical_relevance']:.4f}")
        print(f"Inference Time: {final_results['inference_stats']['avg_inference_time_ms']:.2f}ms")

if __name__ == "__main__":
    main() 