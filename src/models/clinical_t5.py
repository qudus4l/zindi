"""Clinical T5 model implementation for medical text generation.

This module implements a T5-based model fine-tuned for clinical decision
support with safety features and edge optimization capabilities.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClinicalT5Config:
    """Configuration for Clinical T5 model.
    
    Attributes:
        model_name: Base T5 model name
        max_input_length: Maximum input sequence length
        max_output_length: Maximum output sequence length
        num_beams: Number of beams for beam search
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        repetition_penalty: Penalty for repetition
        length_penalty: Length penalty for beam search
        early_stopping: Whether to stop when all beams are finished
        use_cache: Whether to use key-value cache
        confidence_threshold: Minimum confidence for predictions
    """
    model_name: str = "t5-small"
    max_input_length: int = 512
    max_output_length: int = 200
    num_beams: int = 4
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    length_penalty: float = 1.0
    early_stopping: bool = True
    use_cache: bool = True
    confidence_threshold: float = 0.7


class ClinicalT5Model(nn.Module):
    """T5 model adapted for clinical text generation.
    
    This model wraps a pre-trained T5 model with additional features
    for clinical safety and edge deployment optimization.
    """
    
    def __init__(self, config: ClinicalT5Config):
        """Initialize the Clinical T5 model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Initialize base T5 model (will be loaded in setup)
        self.model = None
        self.tokenizer = None
        
        # Clinical safety layers
        self.safety_classifier = nn.Sequential(
            nn.Linear(512, 256),  # T5-small hidden size is 512
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)  # Safe/Unsafe classification
        )
        
        # Confidence estimation layer
        self.confidence_estimator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Medical term attention layer
        self.medical_attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=0.1
        )
        
    def setup_model(self, model_path: Optional[str] = None):
        """Setup the T5 model and tokenizer.
        
        Args:
            model_path: Path to pre-trained model (if None, uses default)
        """
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            
            if model_path:
                logger.info(f"Loading model from {model_path}")
                self.model = T5ForConditionalGeneration.from_pretrained(model_path)
                self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            else:
                logger.info(f"Loading base model: {self.config.model_name}")
                self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_name)
                self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
                
            # Add medical tokens to vocabulary
            self._add_medical_tokens()
            
        except ImportError:
            logger.error("Transformers library not installed. Please install with: pip install transformers")
            raise
    
    def _add_medical_tokens(self):
        """Add medical-specific tokens to the tokenizer."""
        medical_tokens = [
            "<diagnosis>", "</diagnosis>",
            "<treatment>", "</treatment>",
            "<assessment>", "</assessment>",
            "<medication>", "</medication>",
            "<referral>", "</referral>",
            "<education>", "</education>"
        ]
        
        num_added = self.tokenizer.add_special_tokens({
            'additional_special_tokens': medical_tokens
        })
        
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Added {num_added} medical tokens to vocabulary")
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                return_dict: bool = True) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels for training
            return_dict: Whether to return a dictionary
            
        Returns:
            Dict[str, torch.Tensor]: Model outputs including loss if labels provided
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call setup_model() first.")
        
        # Get T5 outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            output_hidden_states=True
        )
        
        # Extract encoder hidden states for safety checks
        encoder_hidden_states = outputs.encoder_last_hidden_state
        
        # Pool encoder states for classification
        pooled_output = encoder_hidden_states.mean(dim=1)
        
        # Safety classification
        safety_logits = self.safety_classifier(pooled_output)
        
        # Confidence estimation
        confidence_scores = self.confidence_estimator(pooled_output)
        
        # Prepare output dictionary
        output_dict = {
            'loss': outputs.loss if labels is not None else None,
            'logits': outputs.logits,
            'safety_logits': safety_logits,
            'confidence_scores': confidence_scores,
            'encoder_hidden_states': encoder_hidden_states
        }
        
        return output_dict
    
    def generate_response(self, prompt: str, 
                         max_length: Optional[int] = None,
                         return_confidence: bool = True) -> Dict[str, Any]:
        """Generate a clinical response for a given prompt.
        
        Args:
            prompt: Input clinical vignette
            max_length: Maximum length of generated response
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dict[str, Any]: Generated response with metadata
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call setup_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            max_length=self.config.max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate with safety checks
        with torch.no_grad():
            # Get encoder outputs for safety check
            encoder_outputs = self.model.encoder(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
            # Safety check
            pooled = encoder_outputs.last_hidden_state.mean(dim=1)
            safety_logits = self.safety_classifier(pooled)
            safety_score = torch.softmax(safety_logits, dim=-1)[0, 1].item()  # Safe class probability
            
            # Confidence estimation
            confidence = self.confidence_estimator(pooled)[0, 0].item()
            
            # Generate response if safe
            if safety_score > 0.5:  # Threshold for safety
                generated_ids = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length or self.config.max_output_length,
                    num_beams=self.config.num_beams,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    repetition_penalty=self.config.repetition_penalty,
                    length_penalty=self.config.length_penalty,
                    early_stopping=self.config.early_stopping,
                    use_cache=self.config.use_cache,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode response
                response = self.tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
            else:
                response = "I need to consult with a senior medical professional for this case."
                confidence = 0.0
        
        result = {
            'response': response,
            'safety_score': safety_score,
            'requires_review': confidence < self.config.confidence_threshold
        }
        
        if return_confidence:
            result['confidence'] = confidence
        
        return result
    
    def prepare_for_quantization(self):
        """Prepare model for quantization."""
        # Set model to evaluation mode
        self.eval()
        
        # Fuse modules where possible
        if hasattr(self.model, 'encoder'):
            self.model.encoder = torch.quantization.fuse_modules(
                self.model.encoder,
                [['dropout', 'layer_norm']]
            )
        
        logger.info("Model prepared for quantization")
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Get number of model parameters.
        
        Args:
            trainable_only: Whether to count only trainable parameters
            
        Returns:
            int: Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def save_pretrained(self, save_path: str):
        """Save model and tokenizer.
        
        Args:
            save_path: Directory to save model
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized.")
        
        # Save T5 model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save additional components
        torch.save({
            'safety_classifier': self.safety_classifier.state_dict(),
            'confidence_estimator': self.confidence_estimator.state_dict(),
            'medical_attention': self.medical_attention.state_dict(),
            'config': self.config
        }, f"{save_path}/clinical_components.pt")
        
        logger.info(f"Model saved to {save_path}")
    
    def load_pretrained(self, load_path: str):
        """Load model and tokenizer.
        
        Args:
            load_path: Directory to load model from
        """
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
        # Load T5 model and tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(load_path)
        self.tokenizer = T5Tokenizer.from_pretrained(load_path)
        
        # Load additional components
        checkpoint = torch.load(f"{load_path}/clinical_components.pt")
        self.safety_classifier.load_state_dict(checkpoint['safety_classifier'])
        self.confidence_estimator.load_state_dict(checkpoint['confidence_estimator'])
        self.medical_attention.load_state_dict(checkpoint['medical_attention'])
        self.config = checkpoint['config']
        
        logger.info(f"Model loaded from {load_path}") 