"""Data preprocessing module for clinical vignettes.

This module handles all preprocessing steps including text normalization,
medical term standardization, and format preparation for model training.
"""

import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedSample:
    """Container for preprocessed clinical data sample.
    
    Attributes:
        id: Sample identifier
        prompt: Preprocessed clinical vignette
        response: Preprocessed clinician response
        metadata: Additional metadata (county, experience, etc.)
        tokens: Tokenized representation (if available)
    """
    id: str
    prompt: str
    response: str
    metadata: Dict[str, Any]
    tokens: Optional[Dict[str, List[int]]] = None


class ClinicalDataPreprocessor:
    """Preprocessor for clinical vignette data.
    
    This class handles all preprocessing steps required to prepare
    clinical data for model training, including normalization,
    standardization, and safety checks.
    """
    
    def __init__(self, config: Any):
        """Initialize the preprocessor.
        
        Args:
            config: Configuration object with preprocessing parameters
        """
        self.config = config
        
        # Medical abbreviations mapping
        self.medical_abbreviations = {
            'bp': 'blood pressure',
            'hr': 'heart rate',
            'rr': 'respiratory rate',
            'temp': 'temperature',
            'spo2': 'oxygen saturation',
            'gcs': 'glasgow coma scale',
            'iv': 'intravenous',
            'im': 'intramuscular',
            'po': 'per oral',
            'prn': 'as needed',
            'stat': 'immediately',
            'bid': 'twice daily',
            'tid': 'three times daily',
            'qid': 'four times daily',
            'uti': 'urinary tract infection',
            'urti': 'upper respiratory tract infection',
            'lrti': 'lower respiratory tract infection',
            'pud': 'peptic ulcer disease',
            'dm': 'diabetes mellitus',
            'htn': 'hypertension',
            'tb': 'tuberculosis',
            'hiv': 'human immunodeficiency virus',
            'ors': 'oral rehydration solution',
            'cbc': 'complete blood count',
            'rbs': 'random blood sugar',
            'ecg': 'electrocardiogram',
            'cxr': 'chest x-ray'
        }
        
        # Kenyan-specific medical terms
        self.kenyan_terms = {
            'dawa': 'medicine',
            'mgonjwa': 'patient',
            'hospitali': 'hospital',
            'daktari': 'doctor',
            'muuguzi': 'nurse'
        }
        
        # Response formatting patterns
        self.response_patterns = {
            'assessment': r'(assessment|diagnosis|impression):\s*',
            'plan': r'(plan|management|treatment):\s*',
            'investigations': r'(investigations?|tests?|labs?):\s*',
            'education': r'(education|counseling|advice):\s*'
        }
        
    def preprocess_dataset(self, df: pd.DataFrame, is_training: bool = True) -> List[PreprocessedSample]:
        """Preprocess entire dataset.
        
        Args:
            df: DataFrame with clinical data
            is_training: Whether this is training data (includes responses)
            
        Returns:
            List[PreprocessedSample]: Preprocessed samples
        """
        logger.info(f"Preprocessing {'training' if is_training else 'test'} dataset with {len(df)} samples")
        
        preprocessed_samples = []
        
        for idx, row in df.iterrows():
            try:
                sample = self._preprocess_sample(row, is_training)
                if sample:
                    preprocessed_samples.append(sample)
            except Exception as e:
                logger.warning(f"Error preprocessing sample {idx}: {e}")
                continue
        
        logger.info(f"Successfully preprocessed {len(preprocessed_samples)} samples")
        
        return preprocessed_samples
    
    def _preprocess_sample(self, row: pd.Series, is_training: bool) -> Optional[PreprocessedSample]:
        """Preprocess a single sample.
        
        Args:
            row: DataFrame row with sample data
            is_training: Whether this is training data
            
        Returns:
            Optional[PreprocessedSample]: Preprocessed sample or None if invalid
        """
        # Extract fields
        sample_id = row.get('Master_Index', row.get('ID_VBWWP', f"sample_{row.name}"))
        prompt = str(row.get('Prompt', ''))
        response = str(row.get('Clinician', '')) if is_training else ''
        
        # Skip if essential fields are missing
        if not prompt:
            return None
        
        # Preprocess prompt
        prompt = self._preprocess_text(prompt)
        prompt = self._standardize_medical_terms(prompt)
        prompt = self._format_prompt(prompt, row)
        
        # Preprocess response if training
        if is_training and response:
            response = self._preprocess_text(response)
            response = self._standardize_medical_terms(response)
            response = self._format_response(response)
        
        # Extract metadata
        metadata = {
            'county': str(row.get('County', '')).lower(),
            'health_level': str(row.get('Health level', '')),
            'years_experience': float(row.get('Years of Experience', 0)) if pd.notna(row.get('Years of Experience')) else None,
            'nursing_competency': str(row.get('Nursing Competency', '')),
            'clinical_panel': str(row.get('Clinical Panel', ''))
        }
        
        return PreprocessedSample(
            id=sample_id,
            prompt=prompt,
            response=response,
            metadata=metadata
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing.
        
        Args:
            text: Raw text
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Fix common encoding issues
        text = text.replace('–', '-').replace('—', '-')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('"', '"').replace('"', '"')
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Remove extra punctuation
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        return text.strip()
    
    def _standardize_medical_terms(self, text: str) -> str:
        """Standardize medical terminology.
        
        Args:
            text: Text with medical terms
            
        Returns:
            str: Text with standardized terms
        """
        # Expand medical abbreviations
        for abbrev, full_form in self.medical_abbreviations.items():
            # Match abbreviation with word boundaries
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, full_form, text, flags=re.IGNORECASE)
        
        # Standardize Kenyan terms
        for kenyan, english in self.kenyan_terms.items():
            pattern = r'\b' + re.escape(kenyan) + r'\b'
            text = re.sub(pattern, f"{kenyan} ({english})", text, flags=re.IGNORECASE)
        
        # Standardize vital signs format
        text = self._standardize_vitals(text)
        
        # Standardize medication dosages
        text = self._standardize_dosages(text)
        
        return text
    
    def _standardize_vitals(self, text: str) -> str:
        """Standardize vital signs format.
        
        Args:
            text: Text containing vital signs
            
        Returns:
            str: Text with standardized vitals
        """
        # Blood pressure: "120/80" -> "blood pressure 120/80"
        text = re.sub(r'\b(\d{2,3})/(\d{2,3})\s*mmhg\b', 
                     r'blood pressure \1/\2 mmhg', text, flags=re.IGNORECASE)
        
        # Temperature: "37.5°C" -> "temperature 37.5 celsius"
        text = re.sub(r'\b(\d{2}\.?\d?)\s*°?c\b', 
                     r'temperature \1 celsius', text, flags=re.IGNORECASE)
        
        # Heart rate: "80bpm" -> "heart rate 80 beats per minute"
        text = re.sub(r'\b(\d{2,3})\s*bpm\b', 
                     r'heart rate \1 beats per minute', text, flags=re.IGNORECASE)
        
        return text
    
    def _standardize_dosages(self, text: str) -> str:
        """Standardize medication dosage formats.
        
        Args:
            text: Text containing dosages
            
        Returns:
            str: Text with standardized dosages
        """
        # mg, g, ml, etc.
        text = re.sub(r'\b(\d+)\s*mg\b', r'\1 milligrams', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(\d+)\s*g\b', r'\1 grams', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(\d+)\s*ml\b', r'\1 milliliters', text, flags=re.IGNORECASE)
        
        return text
    
    def _format_prompt(self, prompt: str, row: pd.Series) -> str:
        """Format prompt with structured information.
        
        Args:
            prompt: Preprocessed prompt text
            row: Original data row
            
        Returns:
            str: Formatted prompt
        """
        # Add context about the nurse
        experience = row.get('Years of Experience')
        if pd.notna(experience):
            context = f"context: nurse with {int(experience)} years experience. "
        else:
            context = "context: nurse. "
        
        # Add facility type if available
        facility = row.get('Health level')
        if pd.notna(facility) and facility:
            context += f"facility: {facility}. "
        
        # Combine context with prompt
        formatted_prompt = context + "case: " + prompt
        
        return formatted_prompt
    
    def _format_response(self, response: str) -> str:
        """Format response for consistency.
        
        Args:
            response: Preprocessed response text
            
        Returns:
            str: Formatted response
        """
        # Remove any section headers for consistent format
        for pattern in self.response_patterns.values():
            response = re.sub(pattern, '', response, flags=re.IGNORECASE)
        
        # Ensure response ends with period
        if response and not response.endswith('.'):
            response += '.'
        
        return response
    
    def prepare_for_training(self, samples: List[PreprocessedSample], 
                           tokenizer: Any = None) -> Dict[str, Any]:
        """Prepare samples for model training.
        
        Args:
            samples: Preprocessed samples
            tokenizer: Optional tokenizer for encoding
            
        Returns:
            Dict[str, Any]: Training-ready data
        """
        prompts = [s.prompt for s in samples]
        responses = [s.response for s in samples]
        
        if tokenizer:
            # Tokenize if tokenizer provided
            encoded_prompts = tokenizer(prompts, truncation=True, 
                                       max_length=self.config.data.max_sequence_length,
                                       padding=True, return_tensors='pt')
            encoded_responses = tokenizer(responses, truncation=True,
                                        max_length=self.config.data.max_response_length,
                                        padding=True, return_tensors='pt')
            
            return {
                'input_ids': encoded_prompts['input_ids'],
                'attention_mask': encoded_prompts['attention_mask'],
                'labels': encoded_responses['input_ids'],
                'raw_prompts': prompts,
                'raw_responses': responses
            }
        else:
            # Return raw text if no tokenizer
            return {
                'prompts': prompts,
                'responses': responses,
                'ids': [s.id for s in samples],
                'metadata': [s.metadata for s in samples]
            }
    
    def save_preprocessed_data(self, samples: List[PreprocessedSample], 
                             output_path: Path) -> None:
        """Save preprocessed data to file.
        
        Args:
            samples: Preprocessed samples
            output_path: Path to save data
        """
        # Convert to DataFrame for easy saving
        data = []
        for sample in samples:
            data.append({
                'id': sample.id,
                'prompt': sample.prompt,
                'response': sample.response,
                **sample.metadata
            })
        
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved preprocessed data to {csv_path}")
        
        # Also save as JSON for flexibility
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved preprocessed data to {json_path}")
    
    def validate_preprocessing(self, samples: List[PreprocessedSample]) -> Dict[str, Any]:
        """Validate preprocessed data quality.
        
        Args:
            samples: Preprocessed samples
            
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {
            'total_samples': len(samples),
            'empty_prompts': sum(1 for s in samples if not s.prompt),
            'empty_responses': sum(1 for s in samples if not s.response),
            'avg_prompt_length': np.mean([len(s.prompt.split()) for s in samples]),
            'avg_response_length': np.mean([len(s.response.split()) for s in samples if s.response]),
            'samples_with_metadata': sum(1 for s in samples if s.metadata.get('years_experience') is not None)
        }
        
        # Check for potential issues
        issues = []
        for i, sample in enumerate(samples):
            if len(sample.prompt.split()) < 10:
                issues.append(f"Sample {sample.id}: Very short prompt")
            if sample.response and len(sample.response.split()) < 5:
                issues.append(f"Sample {sample.id}: Very short response")
        
        validation_results['issues'] = issues[:10]  # First 10 issues
        validation_results['total_issues'] = len(issues)
        
        return validation_results 