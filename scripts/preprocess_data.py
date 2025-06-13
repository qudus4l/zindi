#!/usr/bin/env python3
"""Phase 3: Data Preprocessing Script.

This script preprocesses the clinical vignette data, preparing it for
model training with proper formatting and safety checks.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
import logging
import json

from src.data.loader import ClinicalDataLoader
from src.data.preprocessor import ClinicalDataPreprocessor
from src.utils.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main preprocessing function."""
    logger.info("Starting data preprocessing")
    
    # Load configuration
    config = Config()
    
    # Initialize components
    loader = ClinicalDataLoader(config.data.raw_data_dir)
    preprocessor = ClinicalDataPreprocessor(config)
    
    # Load raw data
    logger.info("Loading raw data")
    train_df = loader.load_train_data()
    test_df = loader.load_test_data()
    
    # Create train/validation split
    train_df, val_df = loader.create_train_val_split(train_df, val_ratio=config.data.val_split)
    
    # Preprocess datasets
    logger.info("Preprocessing training data")
    train_samples = preprocessor.preprocess_dataset(train_df, is_training=True)
    
    logger.info("Preprocessing validation data")
    val_samples = preprocessor.preprocess_dataset(val_df, is_training=True)
    
    logger.info("Preprocessing test data")
    test_samples = preprocessor.preprocess_dataset(test_df, is_training=False)
    
    # Validate preprocessing
    logger.info("Validating preprocessed data")
    train_validation = preprocessor.validate_preprocessing(train_samples)
    val_validation = preprocessor.validate_preprocessing(val_samples)
    
    # Save preprocessed data
    logger.info("Saving preprocessed data")
    preprocessor.save_preprocessed_data(train_samples, config.data.processed_data_dir / "train_processed")
    preprocessor.save_preprocessed_data(val_samples, config.data.processed_data_dir / "val_processed")
    preprocessor.save_preprocessed_data(test_samples, config.data.processed_data_dir / "test_processed")
    
    # Save preprocessing statistics
    stats = {
        'train': {
            'samples': len(train_samples),
            'validation': train_validation
        },
        'validation': {
            'samples': len(val_samples),
            'validation': val_validation
        },
        'test': {
            'samples': len(test_samples)
        }
    }
    
    stats_path = config.data.processed_data_dir / "preprocessing_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("DATA PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print(f"\nAverage prompt length: {train_validation['avg_prompt_length']:.1f} words")
    print(f"Average response length: {train_validation['avg_response_length']:.1f} words")
    print(f"\nPreprocessed data saved to: {config.data.processed_data_dir}")
    print("=" * 70)
    
    # Show sample preprocessed data
    if train_samples:
        print("\nSample preprocessed prompt:")
        print("-" * 30)
        print(train_samples[0].prompt[:200] + "...")
        print("\nSample preprocessed response:")
        print("-" * 30)
        print(train_samples[0].response[:200] + "...")


if __name__ == "__main__":
    main() 