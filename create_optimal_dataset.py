#!/usr/bin/env python3
"""
Create an optimized dataset with high-quality samples for better training.
"""

import json
import numpy as np
from pathlib import Path

def filter_optimal_samples():
    """Filter training data for optimal samples."""
    print("🔍 FILTERING OPTIMAL TRAINING SAMPLES")
    print("=" * 50)
    
    # Load training data
    with open("data/processed/train_processed.json") as f:
        train_data = json.load(f)
    
    print(f"Original training samples: {len(train_data)}")
    
    optimal_samples = []
    
    for item in train_data:
        response = item['response']
        response_len = len(response.split())
        
        # Quality criteria
        is_good_length = 80 <= response_len <= 150  # Optimal length range
        has_structure = any(keyword in response.lower() for keyword in [
            'summary', 'management', 'diagnosis', 'treatment', 'assessment'
        ])
        has_medical_terms = any(term in response.lower() for term in [
            'patient', 'medication', 'symptoms', 'investigation', 'referral'
        ])
        not_too_short = response_len >= 50
        not_repetitive = len(set(response.lower().split())) / len(response.split()) > 0.7
        
        # Score the sample
        quality_score = sum([
            is_good_length * 2,    # Length is most important
            has_structure * 1.5,   # Structure is important
            has_medical_terms * 1, # Medical content
            not_too_short * 1,     # Minimum length
            not_repetitive * 0.5   # Avoid repetition
        ])
        
        if quality_score >= 3.0:  # Threshold for inclusion
            optimal_samples.append(item)
    
    print(f"Optimal samples selected: {len(optimal_samples)} ({len(optimal_samples)/len(train_data)*100:.1f}%)")
    
    # Save optimal dataset
    optimal_path = "data/processed/train_optimal.json"
    with open(optimal_path, 'w') as f:
        json.dump(optimal_samples, f, indent=2)
    
    print(f"✅ Optimal dataset saved to {optimal_path}")
    
    # Analyze improvements
    orig_lengths = [len(item['response'].split()) for item in train_data]
    opt_lengths = [len(item['response'].split()) for item in optimal_samples]
    
    print(f"\nQuality Improvements:")
    print(f"  Average length: {np.mean(orig_lengths):.1f} → {np.mean(opt_lengths):.1f} words")
    print(f"  Length std: {np.std(orig_lengths):.1f} → {np.std(opt_lengths):.1f}")
    
    return optimal_path

if __name__ == "__main__":
    filter_optimal_samples() 