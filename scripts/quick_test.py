#!/usr/bin/env python3
"""Quick test to verify model runs on MPS."""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

def test_mps():
    """Test if model runs on MPS."""
    print("Testing T5 on MPS...")
    
    # Check device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load small model
    print("Loading T5-small...")
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    
    # Move to device
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    # Test inference
    text = "Clinical assessment: Patient presents with fever and cough."
    inputs = tokenizer(f"summarize: {text}", return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print("Running inference...")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Result: {result}")
    print("✓ Test passed!")

if __name__ == "__main__":
    test_mps() 