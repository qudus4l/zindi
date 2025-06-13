#!/usr/bin/env python3
"""
Diagnostic script to identify issues with training and evaluation.
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import torch
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.models.clinical_t5 import ClinicalT5Model
from src.training.trainer import ClinicalDataset
from src.evaluation.metrics import ClinicalMetricsEvaluator
from src.utils.config import Config

def test_data_loading():
    """Test if data is loaded correctly."""
    print("=" * 60)
    print("TESTING DATA LOADING")
    print("=" * 60)
    
    # Load processed data
    train_path = Path("data/processed/train_processed.json")
    val_path = Path("data/processed/val_processed.json")
    
    if not train_path.exists():
        print(f"❌ Training data not found: {train_path}")
        return False
    
    if not val_path.exists():
        print(f"❌ Validation data not found: {val_path}")
        return False
    
    # Load and inspect data
    with open(train_path) as f:
        train_data = json.load(f)
    
    with open(val_path) as f:
        val_data = json.load(f)
    
    print(f"✅ Training samples: {len(train_data)}")
    print(f"✅ Validation samples: {len(val_data)}")
    
    # Check sample structure
    sample = train_data[0]
    print(f"\nSample structure:")
    print(f"  - ID: {sample.get('id', 'MISSING')}")
    print(f"  - Prompt length: {len(sample.get('prompt', ''))}")
    print(f"  - Response length: {len(sample.get('response', ''))}")
    
    # Check response lengths
    response_lengths = [len(item['response'].split()) for item in train_data[:10]]
    print(f"\nFirst 10 response lengths: {response_lengths}")
    print(f"Average response length: {sum(response_lengths) / len(response_lengths):.1f} words")
    
    return True

def test_model_loading():
    """Test if model loads correctly."""
    print("\n" + "=" * 60)
    print("TESTING MODEL LOADING")
    print("=" * 60)
    
    try:
        from src.models.clinical_t5 import ClinicalT5Config
        
        # Create compatible config
        model_config = ClinicalT5Config(model_name="t5-small")
        model = ClinicalT5Model(model_config)
        model.setup_model()  # This is required!
        print(f"✅ Model loaded successfully")
        print(f"✅ Model parameters: {sum(p.numel() for p in model.model.parameters()):,}")
        
        # Test tokenization
        test_prompt = "context: nurse with 5 years experience. case: patient with fever and headache"
        tokens = model.tokenizer(test_prompt, return_tensors="pt", max_length=512, truncation=True, padding=True)
        print(f"✅ Tokenization works: {tokens['input_ids'].shape}")
        
        return model
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None

def test_model_generation(model: ClinicalT5Model):
    """Test model generation capabilities."""
    print("\n" + "=" * 60)
    print("TESTING MODEL GENERATION")
    print("=" * 60)
    
    test_prompts = [
        "context: nurse with 5 years experience. case: patient with fever and headache",
        "context: nurse. case: child with difficulty breathing",
        "context: nurse with 10 years experience. case: elderly patient with chest pain"
    ]
    
    model.model.eval()
    
    for i, prompt in enumerate(test_prompts):
        print(f"\nTest {i+1}:")
        print(f"Input: {prompt[:100]}...")
        
        # Tokenize
        inputs = model.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True, 
            padding=True
        )
        
        # Generate
        with torch.no_grad():
            outputs = model.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=200,
                num_beams=2,
                early_stopping=True,
                do_sample=False
            )
        
        # Decode
        generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Output: {generated_text}")
        print(f"Output length: {len(generated_text.split())} words")

def test_dataset_creation(model: ClinicalT5Model):
    """Test dataset creation and data loading."""
    print("\n" + "=" * 60)
    print("TESTING DATASET CREATION")
    print("=" * 60)
    
    try:
        config = Config()
        
        # Create dataset
        dataset = ClinicalDataset(
            data_path="data/processed/val_processed.json",
            tokenizer=model.tokenizer,
            max_length=512
        )
        
        print(f"✅ Dataset created: {len(dataset)} samples")
        
        # Test data loading
        sample = dataset[0]
        print(f"✅ Sample keys: {list(sample.keys())}")
        print(f"✅ Input shape: {sample['input_ids'].shape}")
        print(f"✅ Labels shape: {sample['labels'].shape}")
        
        # Decode sample
        input_text = model.tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        label_text = model.tokenizer.decode(sample['labels'], skip_special_tokens=True)
        
        print(f"\nSample input: {input_text[:200]}...")
        print(f"Sample label: {label_text[:200]}...")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        return None

def test_evaluation_metrics():
    """Test evaluation metrics."""
    print("\n" + "=" * 60)
    print("TESTING EVALUATION METRICS")
    print("=" * 60)
    
    # Test with simple examples
    predictions = [
        "The patient needs immediate care and monitoring.",
        "Administer oxygen and check vital signs.",
        "Refer to specialist for further evaluation."
    ]
    
    references = [
        "Patient requires immediate medical attention and continuous monitoring.",
        "Provide oxygen therapy and monitor vital signs closely.",
        "Refer patient to specialist for comprehensive evaluation."
    ]
    
    evaluator = ClinicalMetricsEvaluator()
    
    try:
        results = evaluator.evaluate_batch(predictions, references)
        print("✅ Evaluation completed successfully")
        print(f"✅ ROUGE-1 F1: {results['summary']['rouge1_f1']:.4f}")
        print(f"✅ Clinical Relevance: {results['summary']['clinical_relevance']:.4f}")
        
        # Test with very short responses (like our model)
        short_predictions = ["care", "oxygen", "refer"]
        short_results = evaluator.evaluate_batch(short_predictions, references)
        print(f"\n📊 Short responses ROUGE-1 F1: {short_results['summary']['rouge1_f1']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False

def test_actual_model_predictions(model: ClinicalT5Model, dataset: ClinicalDataset):
    """Test what the actual trained model is generating."""
    print("\n" + "=" * 60)
    print("TESTING ACTUAL MODEL PREDICTIONS")
    print("=" * 60)
    
    model.model.eval()
    
    # Test on a few samples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        
        print(f"\n--- Sample {i+1} ---")
        
        # Get input and expected output
        input_text = model.tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        expected_text = model.tokenizer.decode(sample['labels'], skip_special_tokens=True)
        
        print(f"Input: {input_text[:200]}...")
        print(f"Expected: {expected_text[:200]}...")
        
        # Generate prediction
        with torch.no_grad():
            outputs = model.model.generate(
                input_ids=sample['input_ids'].unsqueeze(0),
                attention_mask=sample['attention_mask'].unsqueeze(0),
                max_length=200,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )
        
        predicted_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Predicted: {predicted_text}")
        print(f"Predicted length: {len(predicted_text.split())} words")
        
        # Quick ROUGE calculation
        evaluator = ClinicalMetricsEvaluator()
        rouge_scores = evaluator.calculate_rouge_scores([predicted_text], [expected_text])
        print(f"ROUGE-1 F1: {rouge_scores.rouge1['f1']:.4f}")

def test_checkpoint_loading():
    """Test loading from checkpoint."""
    print("\n" + "=" * 60)
    print("TESTING CHECKPOINT LOADING")
    print("=" * 60)
    
    checkpoint_path = "checkpoints/epoch_0"
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        from src.models.clinical_t5 import ClinicalT5Config
        
        model_config = ClinicalT5Config(model_name="t5-small")
        model = ClinicalT5Model(model_config)
        model.setup_model()  # Setup base model first
        model.load_pretrained(checkpoint_path)
        print(f"✅ Checkpoint loaded successfully")
        return model
        
    except Exception as e:
        print(f"❌ Checkpoint loading failed: {e}")
        return None

def main():
    """Run all diagnostic tests."""
    print("🔍 CLINICAL MODEL DIAGNOSTIC SUITE")
    print("=" * 60)
    
    # Test 1: Data loading
    if not test_data_loading():
        print("❌ Data loading failed - stopping diagnostics")
        return
    
    # Test 2: Model loading
    model = test_model_loading()
    if model is None:
        print("❌ Model loading failed - stopping diagnostics")
        return
    
    # Test 3: Model generation (fresh model)
    test_model_generation(model)
    
    # Test 4: Dataset creation
    dataset = test_dataset_creation(model)
    if dataset is None:
        print("❌ Dataset creation failed - stopping diagnostics")
        return
    
    # Test 5: Evaluation metrics
    if not test_evaluation_metrics():
        print("❌ Evaluation failed - stopping diagnostics")
        return
    
    # Test 6: Checkpoint loading
    trained_model = test_checkpoint_loading()
    if trained_model is not None:
        print("\n🔍 Testing trained model predictions...")
        test_actual_model_predictions(trained_model, dataset)
    
    print("\n" + "=" * 60)
    print("🏁 DIAGNOSTIC COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main() 