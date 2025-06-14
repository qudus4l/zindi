#!/usr/bin/env python3
"""
Advanced training script to achieve FANTASTIC clinical model results.
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import argparse
import torch
import json
import numpy as np
from pathlib import Path
import logging
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.models.clinical_t5 import ClinicalT5Model, ClinicalT5Config
from src.training.trainer import ClinicalDataset, ClinicalTrainer
from src.evaluation.metrics import ClinicalMetricsEvaluator
from src.utils.config import Config

def create_enhanced_dataset():
    """Create enhanced dataset with medical knowledge augmentation."""
    print("🧠 CREATING ENHANCED DATASET WITH MEDICAL KNOWLEDGE")
    print("=" * 60)
    
    # Load optimal training data
    with open("data/processed/train_optimal.json") as f:
        train_data = json.load(f)
    
    enhanced_data = []
    medical_templates = [
        "provide comprehensive clinical assessment and management for:",
        "analyze clinical presentation and recommend treatment for:",
        "evaluate patient condition and suggest management plan for:",
        "assess clinical case and provide detailed care plan for:",
        "review clinical scenario and recommend appropriate intervention for:"
    ]
    
    for item in train_data:
        # Original sample
        enhanced_data.append(item)
        
        # Create enhanced versions with different medical prompts
        prompt_parts = item['prompt'].split('case:')
        if len(prompt_parts) > 1:
            case_description = prompt_parts[1].strip()
            
            # Add 2 enhanced versions per original sample
            for i in range(2):
                template = random.choice(medical_templates)
                enhanced_item = {
                    'id': f"{item['id']}_enhanced_{i}",
                    'prompt': f"context: experienced clinician. case: {template} {case_description}",
                    'response': item['response'],
                    'county': item.get('county', ''),
                    'health_level': item.get('health_level', ''),
                    'years_experience': item.get('years_experience', 0),
                    'nursing_competency': item.get('nursing_competency', ''),
                    'clinical_panel': item.get('clinical_panel', '')
                }
                enhanced_data.append(enhanced_item)
    
    # Save enhanced dataset
    enhanced_path = "data/processed/train_enhanced.json"
    with open(enhanced_path, 'w') as f:
        json.dump(enhanced_data, f, indent=2)
    
    print(f"✅ Enhanced dataset created: {len(train_data)} → {len(enhanced_data)} samples")
    print(f"✅ Saved to: {enhanced_path}")
    
    return enhanced_path

def train_with_advanced_techniques(epochs=50, batch_size=16):
    """Train with advanced techniques for fantastic results."""
    print("🚀 ADVANCED TRAINING FOR FANTASTIC RESULTS")
    print("=" * 60)
    
    # Create enhanced dataset
    enhanced_data_path = create_enhanced_dataset()
    
    # Setup model
    model_config = ClinicalT5Config(model_name="t5-small")
    model = ClinicalT5Model(model_config)
    model.setup_model()
    
    # Load existing checkpoint if available
    if Path("checkpoints/epoch_0").exists():
        logger.info("Loading existing checkpoint")
        model.load_pretrained("checkpoints/epoch_0")
    
    # Create datasets
    train_dataset = ClinicalDataset(
        data_path=enhanced_data_path,
        tokenizer=model.tokenizer,
        max_length=512
    )
    
    val_dataset = ClinicalDataset(
        data_path="data/processed/val_processed.json",
        tokenizer=model.tokenizer,
        max_length=512
    )
    
    # Advanced training configuration
    config = Config()
    config.training.num_epochs = epochs
    config.training.batch_size = batch_size
    config.training.learning_rate = 2e-5  # Lower for fine-tuning
    config.training.warmup_steps = 500    # More warmup
    config.training.early_stopping_patience = 15  # More patience
    config.training.gradient_clip_norm = 0.3      # Tighter clipping
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Train
    trainer = ClinicalTrainer(model, config)
    results = trainer.train(train_dataset, val_dataset)
    
    return results, model, val_dataset

def evaluate_fantastic_model(model, val_dataset):
    """Evaluate model with multiple generation strategies."""
    print("🎯 FANTASTIC MODEL EVALUATION")
    print("=" * 60)
    
    model.eval()
    evaluator = ClinicalMetricsEvaluator()
    
    # Test multiple generation strategies
    strategies = [
        {
            "name": "Balanced",
            "params": {"max_new_tokens": 180, "num_beams": 3, "repetition_penalty": 1.2, "length_penalty": 1.1}
        },
        {
            "name": "Quality",
            "params": {"max_new_tokens": 200, "num_beams": 4, "repetition_penalty": 1.1, "length_penalty": 1.2}
        },
        {
            "name": "Speed",
            "params": {"max_new_tokens": 150, "num_beams": 2, "repetition_penalty": 1.3, "length_penalty": 1.0}
        }
    ]
    
    best_results = None
    best_strategy = None
    
    from torch.utils.data import DataLoader
    import time
    
    for strategy in strategies:
        print(f"\n🔄 Testing {strategy['name']} strategy...")
        
        predictions = []
        references = []
        inference_times = []
        
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                device = next(model.model.parameters()).device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Generate predictions
                start_time = time.time()
                
                generated_ids = model.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    min_length=30,
                    early_stopping=True,
                    do_sample=False,
                    **strategy['params']
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
        
        # Evaluate this strategy
        results = evaluator.evaluate_batch(predictions, references)
        avg_inference_time = sum(inference_times) / len(inference_times)
        
        results['inference_stats'] = {
            'avg_inference_time_ms': avg_inference_time,
            'meets_constraint': avg_inference_time < 100
        }
        
        print(f"  ROUGE-1 F1: {results['summary']['rouge1_f1']:.4f}")
        print(f"  Clinical Relevance: {results['summary']['clinical_relevance']:.4f}")
        print(f"  Inference Time: {avg_inference_time:.2f}ms")
        print(f"  Meets Constraint: {results['inference_stats']['meets_constraint']}")
        
        # Select best strategy (prioritize ROUGE-1 F1 if meets time constraint)
        if results['inference_stats']['meets_constraint']:
            if best_results is None or results['summary']['rouge1_f1'] > best_results['summary']['rouge1_f1']:
                best_results = results
                best_strategy = strategy['name']
        elif best_results is None:  # Fallback if no strategy meets constraint
            best_results = results
            best_strategy = strategy['name']
    
    # Print final results
    print("\n" + "=" * 70)
    print(f"🏆 BEST STRATEGY: {best_strategy}")
    print("=" * 70)
    print(evaluator.format_results(best_results))
    print(f"\nInference Time: {best_results['inference_stats']['avg_inference_time_ms']:.2f}ms")
    print(f"Meets <100ms constraint: {best_results['inference_stats']['meets_constraint']}")
    
    # Fantastic results check
    rouge1_f1 = best_results['summary']['rouge1_f1']
    clinical_relevance = best_results['summary']['clinical_relevance']
    
    print("\n🎯 FANTASTIC RESULTS CHECK:")
    print(f"  ROUGE-1 F1: {rouge1_f1:.4f} {'✅' if rouge1_f1 >= 0.50 else '❌'} (Target: ≥0.50)")
    print(f"  Clinical Relevance: {clinical_relevance:.4f} {'✅' if clinical_relevance >= 0.70 else '❌'} (Target: ≥0.70)")
    print(f"  Speed Constraint: {'✅' if best_results['inference_stats']['meets_constraint'] else '❌'} (Target: <100ms)")
    
    if rouge1_f1 >= 0.50 and clinical_relevance >= 0.70 and best_results['inference_stats']['meets_constraint']:
        print("\n🎉 FANTASTIC RESULTS ACHIEVED! 🎉")
    elif rouge1_f1 >= 0.45 and clinical_relevance >= 0.60:
        print("\n🎯 EXCELLENT RESULTS! Close to fantastic!")
    else:
        print("\n📈 GOOD RESULTS! Room for improvement.")
    
    print("=" * 70)
    
    return best_results

def main():
    """Main function for fantastic training."""
    parser = argparse.ArgumentParser(description="Train for fantastic results")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate')
    
    args = parser.parse_args()
    
    if args.eval_only:
        # Setup model and evaluate
        model_config = ClinicalT5Config(model_name="t5-small")
        model = ClinicalT5Model(model_config)
        model.setup_model()
        
        if Path("checkpoints/epoch_0").exists():
            model.load_pretrained("checkpoints/epoch_0")
        
        val_dataset = ClinicalDataset(
            data_path="data/processed/val_processed.json",
            tokenizer=model.tokenizer,
            max_length=512
        )
        
        evaluate_fantastic_model(model, val_dataset)
    else:
        # Full training
        results, model, val_dataset = train_with_advanced_techniques(args.epochs, args.batch_size)
        
        # Final evaluation
        final_results = evaluate_fantastic_model(model, val_dataset)
        
        print(f"\n🎯 TRAINING COMPLETE!")
        print(f"Best ROUGE-1 F1: {final_results['summary']['rouge1_f1']:.4f}")
        print(f"Clinical Relevance: {final_results['summary']['clinical_relevance']:.4f}")

if __name__ == "__main__":
    main() 