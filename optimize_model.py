#!/usr/bin/env python3
"""
Comprehensive model optimization script to achieve fantastic results.
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import torch
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import logging
import time
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.models.clinical_t5 import ClinicalT5Model, ClinicalT5Config
from src.training.trainer import ClinicalDataset
from src.evaluation.metrics import ClinicalMetricsEvaluator
from src.utils.config import Config

def analyze_current_performance():
    """Analyze current model performance in detail."""
    print("🔍 ANALYZING CURRENT MODEL PERFORMANCE")
    print("=" * 60)
    
    # Load the best model
    checkpoint_path = "checkpoints/epoch_0"
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    # Setup model
    model_config = ClinicalT5Config(model_name="t5-small")
    model = ClinicalT5Model(model_config)
    model.setup_model()
    model.load_pretrained(checkpoint_path)
    model.eval()
    
    # Load validation data
    dataset = ClinicalDataset(
        data_path="data/processed/val_processed.json",
        tokenizer=model.tokenizer,
        max_length=512
    )
    
    print(f"✅ Model loaded from {checkpoint_path}")
    print(f"✅ Validation dataset: {len(dataset)} samples")
    
    return model, dataset

def detailed_response_analysis(model, dataset, num_samples=10):
    """Analyze model responses in detail."""
    print("\n📝 DETAILED RESPONSE ANALYSIS")
    print("=" * 60)
    
    evaluator = ClinicalMetricsEvaluator()
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        
        print(f"\n--- Sample {i+1} ---")
        
        # Get input and expected output
        input_text = model.tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        
        # Handle -100 labels properly
        labels_for_decode = sample['labels'].clone()
        labels_for_decode[labels_for_decode == -100] = model.tokenizer.pad_token_id
        expected_text = model.tokenizer.decode(labels_for_decode, skip_special_tokens=True)
        
        print(f"Input: {input_text[:150]}...")
        print(f"Expected: {expected_text[:150]}...")
        
        # Generate prediction with current settings
        start_time = time.time()
        with torch.no_grad():
            outputs = model.model.generate(
                input_ids=sample['input_ids'].unsqueeze(0),
                attention_mask=sample['attention_mask'].unsqueeze(0),
                max_new_tokens=200,
                min_length=20,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
                repetition_penalty=1.2,
                length_penalty=1.0
            )
        
        inference_time = (time.time() - start_time) * 1000
        predicted_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Predicted: {predicted_text}")
        print(f"Predicted length: {len(predicted_text.split())} words")
        print(f"Inference time: {inference_time:.2f}ms")
        
        # Calculate ROUGE for this sample
        rouge_scores = evaluator.calculate_rouge_scores([predicted_text], [expected_text])
        print(f"ROUGE-1 F1: {rouge_scores.rouge1['f1']:.4f}")
        
        # Analyze issues
        issues = []
        if len(predicted_text.split()) < 50:
            issues.append("Too short")
        if rouge_scores.rouge1['f1'] < 0.3:
            issues.append("Low ROUGE score")
        if inference_time > 100:
            issues.append("Slow inference")
        
        if issues:
            print(f"⚠️  Issues: {', '.join(issues)}")
        else:
            print("✅ Good response")

def analyze_data_quality():
    """Analyze training data quality for optimization opportunities."""
    print("\n📊 DATA QUALITY ANALYSIS")
    print("=" * 60)
    
    # Load training data
    with open("data/processed/train_processed.json") as f:
        train_data = json.load(f)
    
    # Analyze response characteristics
    response_lengths = [len(item['response'].split()) for item in train_data]
    prompt_lengths = [len(item['prompt'].split()) for item in train_data]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Average response length: {np.mean(response_lengths):.1f} words")
    print(f"Response length std: {np.std(response_lengths):.1f}")
    print(f"Average prompt length: {np.mean(prompt_lengths):.1f} words")
    
    # Find optimal samples (good length, clear structure)
    optimal_samples = []
    for i, item in enumerate(train_data):
        response_len = len(item['response'].split())
        if 80 <= response_len <= 150:  # Good length range
            if 'summary' in item['response'].lower() or 'management' in item['response'].lower():
                optimal_samples.append((i, item, response_len))
    
    print(f"Optimal training samples: {len(optimal_samples)} ({len(optimal_samples)/len(train_data)*100:.1f}%)")
    
    # Analyze medical terminology coverage
    medical_terms = set()
    for item in train_data:
        text = item['response'].lower()
        # Extract medical terms (simplified)
        words = text.split()
        for word in words:
            if any(term in word for term in ['diagnosis', 'treatment', 'medication', 'symptom', 'patient', 'clinical']):
                medical_terms.add(word)
    
    print(f"Unique medical terms: {len(medical_terms)}")
    
    return optimal_samples

def optimize_generation_parameters(model, dataset):
    """Test different generation parameters to optimize performance."""
    print("\n⚙️ OPTIMIZING GENERATION PARAMETERS")
    print("=" * 60)
    
    # Test different parameter combinations
    param_configs = [
        {"name": "Current", "max_new_tokens": 200, "min_length": 20, "num_beams": 4, "repetition_penalty": 1.2},
        {"name": "Faster", "max_new_tokens": 150, "min_length": 30, "num_beams": 2, "repetition_penalty": 1.3},
        {"name": "Quality", "max_new_tokens": 250, "min_length": 40, "num_beams": 6, "repetition_penalty": 1.1},
        {"name": "Balanced", "max_new_tokens": 180, "min_length": 35, "num_beams": 3, "repetition_penalty": 1.25},
    ]
    
    evaluator = ClinicalMetricsEvaluator()
    results = []
    
    # Test on first 5 samples
    test_samples = [dataset[i] for i in range(5)]
    
    for config in param_configs:
        print(f"\nTesting {config['name']} configuration...")
        
        predictions = []
        references = []
        inference_times = []
        
        for sample in test_samples:
            # Generate prediction
            start_time = time.time()
            with torch.no_grad():
                outputs = model.model.generate(
                    input_ids=sample['input_ids'].unsqueeze(0),
                    attention_mask=sample['attention_mask'].unsqueeze(0),
                    max_new_tokens=config['max_new_tokens'],
                    min_length=config['min_length'],
                    num_beams=config['num_beams'],
                    early_stopping=True,
                    do_sample=False,
                    repetition_penalty=config['repetition_penalty'],
                    length_penalty=1.0
                )
            
            inference_time = (time.time() - start_time) * 1000
            inference_times.append(inference_time)
            
            predicted_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(predicted_text)
            
            # Get reference
            labels_for_decode = sample['labels'].clone()
            labels_for_decode[labels_for_decode == -100] = model.tokenizer.pad_token_id
            reference_text = model.tokenizer.decode(labels_for_decode, skip_special_tokens=True)
            references.append(reference_text)
        
        # Evaluate
        rouge_scores = evaluator.calculate_rouge_scores(predictions, references)
        avg_inference_time = np.mean(inference_times)
        avg_length = np.mean([len(p.split()) for p in predictions])
        
        result = {
            'config': config['name'],
            'rouge1_f1': rouge_scores.rouge1['f1'],
            'avg_inference_time': avg_inference_time,
            'avg_length': avg_length,
            'meets_time_constraint': avg_inference_time < 100
        }
        results.append(result)
        
        print(f"  ROUGE-1 F1: {result['rouge1_f1']:.4f}")
        print(f"  Avg inference: {result['avg_inference_time']:.2f}ms")
        print(f"  Avg length: {result['avg_length']:.1f} words")
        print(f"  Meets constraint: {result['meets_time_constraint']}")
    
    # Find best configuration
    best_config = max(results, key=lambda x: x['rouge1_f1'] if x['meets_time_constraint'] else 0)
    print(f"\n🏆 Best configuration: {best_config['config']}")
    
    return results

def suggest_improvements():
    """Suggest specific improvements based on analysis."""
    print("\n💡 IMPROVEMENT RECOMMENDATIONS")
    print("=" * 60)
    
    improvements = [
        {
            "category": "🚀 Inference Speed",
            "suggestions": [
                "Reduce num_beams from 4 to 2-3 for faster generation",
                "Use max_new_tokens=150 instead of 200",
                "Implement model quantization (int8)",
                "Use ONNX conversion for edge deployment"
            ]
        },
        {
            "category": "📈 ROUGE Score Improvement", 
            "suggestions": [
                "Increase training epochs to 30-40",
                "Implement curriculum learning (easy→hard samples)",
                "Add data augmentation with paraphrasing",
                "Fine-tune on high-quality samples only",
                "Implement reinforcement learning with ROUGE reward"
            ]
        },
        {
            "category": "🏥 Clinical Relevance",
            "suggestions": [
                "Add medical terminology embeddings",
                "Implement clinical knowledge distillation",
                "Use medical domain-specific pre-training",
                "Add clinical reasoning chain prompts",
                "Implement medical fact verification"
            ]
        },
        {
            "category": "📊 Data Optimization",
            "suggestions": [
                "Filter training data for optimal length (80-150 words)",
                "Augment data with medical paraphrasing",
                "Add synthetic clinical scenarios",
                "Implement active learning for hard samples",
                "Balance dataset across medical specialties"
            ]
        }
    ]
    
    for improvement in improvements:
        print(f"\n{improvement['category']}:")
        for suggestion in improvement['suggestions']:
            print(f"  • {suggestion}")

def create_optimization_plan():
    """Create a concrete optimization plan."""
    print("\n📋 OPTIMIZATION PLAN")
    print("=" * 60)
    
    plan = [
        {
            "phase": "Phase 1: Quick Wins (1-2 hours)",
            "actions": [
                "Optimize generation parameters (num_beams=2, max_new_tokens=150)",
                "Implement model quantization for speed",
                "Filter training data for optimal samples",
                "Retrain for 5 epochs with optimized data"
            ],
            "expected_improvement": "Inference: <100ms, ROUGE-1: 0.35+"
        },
        {
            "phase": "Phase 2: Quality Improvements (2-4 hours)",
            "actions": [
                "Implement curriculum learning",
                "Add medical terminology augmentation", 
                "Train for 30 epochs with learning rate scheduling",
                "Add clinical reasoning prompts"
            ],
            "expected_improvement": "ROUGE-1: 0.45+, Clinical relevance: 0.7+"
        },
        {
            "phase": "Phase 3: Advanced Optimization (4-6 hours)",
            "actions": [
                "Implement ONNX conversion",
                "Add reinforcement learning with ROUGE reward",
                "Medical knowledge distillation",
                "Ensemble multiple models"
            ],
            "expected_improvement": "ROUGE-1: 0.55+, Inference: <80ms"
        }
    ]
    
    for phase in plan:
        print(f"\n{phase['phase']}:")
        print(f"Expected: {phase['expected_improvement']}")
        for action in phase['actions']:
            print(f"  • {action}")

def main():
    """Run comprehensive optimization analysis."""
    print("🎯 CLINICAL MODEL OPTIMIZATION SUITE")
    print("=" * 60)
    
    # Step 1: Analyze current performance
    model_data = analyze_current_performance()
    if model_data is None:
        return
    
    model, dataset = model_data
    
    # Step 2: Detailed response analysis
    detailed_response_analysis(model, dataset, num_samples=5)
    
    # Step 3: Data quality analysis
    optimal_samples = analyze_data_quality()
    
    # Step 4: Optimize generation parameters
    param_results = optimize_generation_parameters(model, dataset)
    
    # Step 5: Suggest improvements
    suggest_improvements()
    
    # Step 6: Create optimization plan
    create_optimization_plan()
    
    print("\n🏁 OPTIMIZATION ANALYSIS COMPLETE")
    print("=" * 60)
    print("Ready to implement improvements for FANTASTIC results! 🚀")

if __name__ == "__main__":
    main() 