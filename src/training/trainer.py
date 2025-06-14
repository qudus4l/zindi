"""Training module for clinical decision support model.

This module implements the training pipeline with clinical safety
callbacks and performance monitoring.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from pathlib import Path
import json
from tqdm import tqdm
import time
from dataclasses import dataclass

from src.utils.device_utils import get_optimal_device, get_device_info, optimize_for_device, get_memory_info, clear_memory

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    epoch: int
    train_loss: float
    val_loss: float
    rouge_scores: Dict[str, float]
    safety_violations: int
    avg_confidence: float
    learning_rate: float
    time_elapsed: float


class ClinicalDataset(Dataset):
    """Dataset for clinical vignettes and responses."""
    
    def __init__(self, data_path: Path, tokenizer: Any, max_length: int = 512):
        """Initialize the dataset.
        
        Args:
            data_path: Path to preprocessed data file
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict[str, torch.Tensor]: Tokenized inputs and labels
        """
        sample = self.data[idx]
        
        # Prepare input text with clear task instruction for T5
        input_text = f"Clinical case: {sample['prompt']} Provide clinical response:"
        target_text = sample['response']
        
        # Tokenize inputs
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize targets - CRITICAL: Use different max_length for targets
        targets = self.tokenizer(
            target_text,
            max_length=256,  # Shorter for responses to avoid truncation
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # CRITICAL FIX: Properly mask padding tokens in labels
        labels = targets['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding in loss
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels
        }


class ClinicalSafetyCallback:
    """Callback for monitoring clinical safety during training."""
    
    def __init__(self, harmful_patterns: List[str], confidence_threshold: float = 0.7):
        """Initialize the safety callback.
        
        Args:
            harmful_patterns: List of harmful pattern keywords
            confidence_threshold: Minimum acceptable confidence
        """
        self.harmful_patterns = harmful_patterns
        self.confidence_threshold = confidence_threshold
        self.violations = []
    
    def check_output(self, output: str, confidence: float, sample_id: str) -> bool:
        """Check if output is clinically safe.
        
        Args:
            output: Generated text
            confidence: Model confidence score
            sample_id: Sample identifier
            
        Returns:
            bool: True if safe, False if violation detected
        """
        # Check for harmful patterns
        output_lower = output.lower()
        for pattern in self.harmful_patterns:
            if pattern in output_lower:
                self.violations.append({
                    'sample_id': sample_id,
                    'pattern': pattern,
                    'output_snippet': output[:100],
                    'confidence': confidence
                })
                return False
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            self.violations.append({
                'sample_id': sample_id,
                'reason': 'low_confidence',
                'confidence': confidence
            })
            return False
        
        return True
    
    def get_report(self) -> Dict[str, Any]:
        """Get safety violation report.
        
        Returns:
            Dict[str, Any]: Safety report
        """
        return {
            'total_violations': len(self.violations),
            'violation_types': self._categorize_violations(),
            'recent_violations': self.violations[-10:]  # Last 10 violations
        }
    
    def _categorize_violations(self) -> Dict[str, int]:
        """Categorize violations by type."""
        categories = {}
        for violation in self.violations:
            if 'pattern' in violation:
                category = f"harmful_pattern_{violation['pattern']}"
            else:
                category = violation.get('reason', 'unknown')
            
            categories[category] = categories.get(category, 0) + 1
        
        return categories


class ClinicalTrainer:
    """Trainer for clinical decision support model."""
    
    def __init__(self, model: nn.Module, config: Any, device: Optional[str] = None):
        """Initialize the trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            device: Device to use for training (optional, uses config if not provided)
        """
        self.model = model
        self.config = config
        
        # Device selection using utility function
        preferred_device = device or config.training.device
        self.device = get_optimal_device(preferred_device)
        
        # Log device information
        device_info = get_device_info()
        logger.info(f"Device information: {device_info}")
        
        # Move model to device
        self.model.to(self.device)
        
        # Log memory info if available
        memory_info = get_memory_info(self.device)
        if memory_info:
            logger.info(f"Device memory: {memory_info['total']:.2f}GB total, "
                       f"{memory_info['allocated']:.2f}GB allocated")
        
        # Initialize components
        self.optimizer = None
        self.scheduler = None
        
        # Mixed precision setup based on device
        if config.training.use_mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled (CUDA)")
        else:
            self.scaler = None
            if config.training.use_mixed_precision and self.device.type != 'cuda':
                logger.warning(f"Mixed precision not supported on {self.device.type}, disabling")
        
        # Safety callback
        self.safety_callback = ClinicalSafetyCallback(
            harmful_patterns=['stop all medication', 'ignore symptoms', 'no need for doctor'],
            confidence_threshold=config.clinical_safety.confidence_threshold
        )
        
        # Metrics tracking
        self.training_history = []
        self.best_rouge_score = 0.0
        self.best_model_path = None
        
    def setup_optimization(self, steps_per_epoch: Optional[int] = None):
        """Setup optimizer and scheduler."""
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.training.learning_rate,
            weight_decay=0.01
        )
        
        # Setup scheduler only if we have steps_per_epoch
        if steps_per_epoch is not None:
            self.steps_per_epoch = steps_per_epoch
            total_steps = self.config.training.num_epochs * self.steps_per_epoch
            warmup_steps = self.config.training.warmup_steps
            
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
            
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            logger.info(f"Optimizer setup complete. Total steps: {total_steps}, Warmup: {warmup_steps}")
        else:
            self.scheduler = None
            logger.info("Optimizer setup complete. Scheduler will be set up during training.")
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset, 
              num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of epochs (overrides config if provided)
            
        Returns:
            Dict[str, Any]: Training results
        """
        num_epochs = num_epochs or self.config.training.num_epochs
        
        # Get device-optimized settings
        device_config = optimize_for_device(self.device, {
            'use_mixed_precision': self.config.training.use_mixed_precision,
            'pin_memory': False,
            'num_workers': 0
        })
        
        # Create data loaders with device-optimized settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=device_config['num_workers'],
            pin_memory=device_config['pin_memory']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=device_config['num_workers'],
            pin_memory=device_config['pin_memory']
        )
        
        # Setup optimization with steps per epoch
        self.setup_optimization(steps_per_epoch=len(train_loader))
        
        # Training loop
        logger.info(f"Starting training for {num_epochs} epochs on {self.device}")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            val_metrics = self._validate(val_loader, epoch)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            epoch_time = time.time() - epoch_start_time
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_metrics['loss'],
                rouge_scores=val_metrics['rouge_scores'],
                safety_violations=len(self.safety_callback.violations),
                avg_confidence=val_metrics['avg_confidence'],
                learning_rate=self.optimizer.param_groups[0]['lr'],
                time_elapsed=epoch_time
            )
            
            self.training_history.append(metrics)
            self._log_metrics(metrics)
            
            # Save best model
            if val_metrics['rouge_scores']['rouge1']['f1'] > self.best_rouge_score:
                self.best_rouge_score = val_metrics['rouge_scores']['rouge1']['f1']
                self._save_checkpoint(epoch, metrics)
            
            # Clear memory cache periodically
            if epoch % 5 == 0:
                clear_memory(self.device)
            
            # Early stopping check
            if self._should_stop_early():
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Training complete
        results = {
            'training_history': self.training_history,
            'best_rouge_score': self.best_rouge_score,
            'best_model_path': self.best_model_path,
            'safety_report': self.safety_callback.get_report(),
            'device_used': str(self.device),
            'final_memory_info': get_memory_info(self.device)
        }
        
        return results
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass - CRITICAL FIX: Use T5 model directly
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model.model(  # Use the T5 model directly
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    loss = outputs.loss
            else:
                outputs = self.model.model(  # Use the T5 model directly
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                              self.config.training.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                              self.config.training.gradient_clip_norm)
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Clinical validation check (every N steps)
            if num_batches % self.config.training.safety_check_frequency == 0:
                self._perform_safety_check(batch)
        
        return total_loss / num_batches
    
    def _validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, Any]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Dict[str, Any]: Validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_references = []
        confidence_scores = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass - Use T5 model directly
                outputs = self.model.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                total_loss += outputs.loss.item()
                
                # Generate predictions for ROUGE evaluation - OPTIMIZED
                generated_ids = self.model.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=150,  # Reduced for speed
                    min_length=30,       # Higher minimum for quality
                    num_beams=2,         # Reduced for speed
                    early_stopping=True,
                    do_sample=False,     # Deterministic generation
                    repetition_penalty=1.3,  # Higher to avoid repetition
                    length_penalty=1.1   # Encourage longer responses
                )
                
                # Decode predictions and references - CRITICAL FIX
                predictions = self.model.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                
                # Fix reference decoding - handle -100 labels properly
                labels_for_decode = batch['labels'].clone()
                labels_for_decode[labels_for_decode == -100] = self.model.tokenizer.pad_token_id
                references = self.model.tokenizer.batch_decode(
                    labels_for_decode, skip_special_tokens=True
                )
                
                all_predictions.extend(predictions)
                all_references.extend(references)
                
                # Simple confidence estimation based on loss
                batch_confidence = torch.exp(-outputs.loss).item()
                confidence_scores.extend([batch_confidence] * len(predictions))
        
        # Calculate metrics
        from src.evaluation.metrics import ClinicalMetricsEvaluator
        evaluator = ClinicalMetricsEvaluator()
        
        rouge_scores = evaluator.calculate_rouge_scores(all_predictions, all_references)
        
        return {
            'loss': total_loss / len(val_loader),
            'rouge_scores': {
                'rouge1': rouge_scores.rouge1,
                'rouge2': rouge_scores.rouge2,
                'rougeL': rouge_scores.rougeL
            },
            'avg_confidence': np.mean(confidence_scores)
        }
    
    def _perform_safety_check(self, batch: Dict[str, torch.Tensor]):
        """Perform safety check on batch outputs."""
        # Generate outputs for safety check
        with torch.no_grad():
            # Generate predictions
            generated_ids = self.model.model.generate(
                input_ids=batch['input_ids'][:2],  # Check first 2 samples
                attention_mask=batch['attention_mask'][:2],
                max_new_tokens=100,
                min_length=10,
                num_beams=2,
                early_stopping=True
            )
            
            predictions = self.model.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            
            # Simple confidence estimation
            confidences = [0.8] * len(predictions)  # Default confidence
            
            # Check each output
            for i, (pred, conf) in enumerate(zip(predictions, confidences)):
                sample_id = f"sample_{i}"
                self.safety_callback.check_output(pred, conf, sample_id)
    
    def _save_checkpoint(self, epoch: int, metrics: TrainingMetrics):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.project_root) / "checkpoints" / f"epoch_{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(str(checkpoint_dir))
        
        # Save training state
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }, checkpoint_dir / "training_state.pt")
        
        self.best_model_path = str(checkpoint_dir)
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def _should_stop_early(self) -> bool:
        """Check if training should stop early."""
        if len(self.training_history) < self.config.training.early_stopping_patience + 1:
            return False
        
        # Check if validation loss hasn't improved
        recent_losses = [m.val_loss for m in self.training_history[-self.config.training.early_stopping_patience:]]
        return all(loss >= recent_losses[0] for loss in recent_losses[1:])
    
    def _log_metrics(self, metrics: TrainingMetrics):
        """Log training metrics."""
        logger.info(f"Epoch {metrics.epoch}: "
                   f"Train Loss: {metrics.train_loss:.4f}, "
                   f"Val Loss: {metrics.val_loss:.4f}, "
                   f"ROUGE-1 F1: {metrics.rouge_scores['rouge1']['f1']:.4f}, "
                   f"Confidence: {metrics.avg_confidence:.4f}, "
                   f"LR: {metrics.learning_rate:.6f}")
        
        if metrics.safety_violations > 0:
            logger.warning(f"Safety violations detected: {metrics.safety_violations}") 