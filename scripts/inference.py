#!/usr/bin/env python3
"""Inference script for clinical decision support model."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
import logging
from tqdm import tqdm
import time
import string

from src.models.clinical_t5 import ClinicalT5Model, ClinicalT5Config
from src.data.preprocessor import ClinicalDataPreprocessor
from src.utils.config import Config
from src.utils.device_utils import get_optimal_device, get_device_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_response(response):
    """Format response for submission."""
    # Convert to lowercase
    response = response.lower()
    
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    response = response.translate(translator)
    
    # Replace multiple spaces with single space
    response = ' '.join(response.split())
    
    return response


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--test-file', type=str, default='data/raw/test.csv')
    parser.add_argument('--output-file', type=str, default='submission.csv')
    parser.add_argument('--device', type=str, choices=['cuda', 'mps', 'cpu', 'auto'], 
                        default='auto', help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    # Log device information
    device_info = get_device_info()
    logger.info(f"Available devices: {device_info}")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = ClinicalT5Model(ClinicalT5Config())
    model.load_pretrained(args.model_path)
    model.eval()
    
    device = get_optimal_device(args.device)
    model.to(device)
    logger.info(f"Running inference on device: {device}")
    
    # Load test data
    logger.info(f"Loading test data from {args.test_file}")
    test_df = pd.read_csv(args.test_file)
    
    # Preprocess
    preprocessor = ClinicalDataPreprocessor(config)
    test_samples = preprocessor.preprocess_dataset(test_df, is_training=False)
    
    # Generate predictions
    predictions = []
    
    for sample in tqdm(test_samples, desc="Generating predictions"):
        # Generate response
        result = model.generate_response(sample.prompt)
        
        # Format response
        formatted_response = format_response(result['response'])
        
        predictions.append({
            'ID': sample.id,
            'Clinician response': formatted_response
        })
    
    # Save predictions
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(args.output_file, index=False)
    
    logger.info(f"Predictions saved to {args.output_file}")


if __name__ == "__main__":
    main() 