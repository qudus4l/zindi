#!/usr/bin/env python3
"""
Evaluate the latest trained model with optimized parameters.
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import torch
import time
from pathlib import Path
from torch.utils.data import DataLoader

from src.models.clinical_t5 import ClinicalT5Model, ClinicalT5Config
from src.training.trainer import ClinicalDataset
from src.evaluation.metrics import ClinicalMetricsEvaluator

def evaluate_with_optimized_params():
    """Evaluate model with optimized generation parameters."""
    print("🎯 EVALUATING WITH OPTIMIZED PARAMETERS")
    print("=" * 60)
    
    # Setup model
    model_config = ClinicalT5Config(model_name="t5-small")
    model = ClinicalT5Model(model_config)
    model.setup_model()
    
    # Try to load the best available checkpoint
    checkpoint_paths = [
        "checkpoints/epoch_69",
        "/home/shadeform/zindi/checkpoints/epoch_69", 
        "checkpoints/epoch_0"
    ]
    
    model_loaded = False
    for checkpoint_path in checkpoint_paths:
        if Path(checkpoint_path).exists():
            print(f"Loading model from: {checkpoint_path}")
            model.load_pretrained(checkpoint_path)
            model_loaded = True
            break
    
    if not model_loaded:
        print("❌ No checkpoint found, using base model")
        return
    
    model.eval()
    
    # Load validation dataset
    val_dataset = ClinicalDataset(
        data_path="data/processed/val_processed.json",
        tokenizer=model.tokenizer,
        max_length=512
    )
    
    print(f"✅ Validation dataset: {len(val_dataset)} samples")
    
    # Evaluate with optimized parameters
    evaluator = ClinicalMetricsEvaluator()
    predictions = []
    references = []
    inference_times = []
    
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    print("🔄 Generating predictions with optimized parameters...")
    
    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            device = next(model.model.parameters()).device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Generate with OPTIMIZED parameters
            start_time = time.time()
            
            generated_ids = model.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_new_tokens=150,    # Optimized for speed
                min_length=30,         # Quality minimum
                num_beams=2,           # Speed optimization
                early_stopping=True,
                do_sample=False,
                repetition_penalty=1.3, # Avoid repetition
                length_penalty=1.1     # Encourage longer responses
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
    avg_inference_time = sum(inference_times) / len(inference_times)
    results['inference_stats'] = {
        'avg_inference_time_ms': avg_inference_time,
        'max_inference_time_ms': max(inference_times),
        'meets_constraint': avg_inference_time < 100
    }
    
    # Print results
    print("\n" + "=" * 70)
    print("🎯 OPTIMIZED MODEL EVALUATION RESULTS")
    print("=" * 70)
    print(evaluator.format_results(results))
    print(f"\nInference Time: {avg_inference_time:.2f}ms (avg)")
    print(f"Meets <100ms constraint: {results['inference_stats']['meets_constraint']}")
    print("=" * 70)
    
    # Show sample predictions
    print("\n📝 SAMPLE PREDICTIONS:")
    print("-" * 50)
    for i in range(min(3, len(predictions))):
        print(f"\nSample {i+1}:")
        print(f"Predicted: {predictions[i][:200]}...")
        print(f"Reference: {references[i][:200]}...")
        print(f"Length: {len(predictions[i].split())} words")
    
    return results

if __name__ == "__main__":
    evaluate_with_optimized_params() 