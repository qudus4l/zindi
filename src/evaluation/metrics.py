"""Evaluation metrics for clinical decision support model."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import re
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class RougeScores:
    """Container for ROUGE evaluation scores."""
    rouge1: Dict[str, float]
    rouge2: Dict[str, float]
    rougeL: Dict[str, float]
    rougeLsum: Optional[Dict[str, float]] = None


class ClinicalMetricsEvaluator:
    """Evaluator for clinical text generation metrics."""
    
    def __init__(self):
        """Initialize the metrics evaluator."""
        self.medical_keywords = {
            'assessment', 'diagnosis', 'treatment', 'management',
            'medication', 'monitoring', 'referral', 'education',
            'counseling', 'follow-up', 'investigation', 'examination'
        }
        
    def calculate_rouge_scores(self, predictions: List[str], 
                             references: List[str]) -> RougeScores:
        """Calculate ROUGE scores for predictions."""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        # Initialize score accumulators
        rouge1_scores = {'precision': [], 'recall': [], 'f1': []}
        rouge2_scores = {'precision': [], 'recall': [], 'f1': []}
        rougeL_scores = {'precision': [], 'recall': [], 'f1': []}
        
        for pred, ref in zip(predictions, references):
            # Preprocess texts
            pred_tokens = self._tokenize(pred)
            ref_tokens = self._tokenize(ref)
            
            # Calculate ROUGE-1
            r1_scores = self._calculate_rouge_n(pred_tokens, ref_tokens, n=1)
            for key in rouge1_scores:
                rouge1_scores[key].append(r1_scores[key])
            
            # Calculate ROUGE-2
            r2_scores = self._calculate_rouge_n(pred_tokens, ref_tokens, n=2)
            for key in rouge2_scores:
                rouge2_scores[key].append(r2_scores[key])
            
            # Calculate ROUGE-L
            rl_scores = self._calculate_rouge_l(pred_tokens, ref_tokens)
            for key in rougeL_scores:
                rougeL_scores[key].append(rl_scores[key])
        
        # Average scores
        rouge_scores = RougeScores(
            rouge1={k: np.mean(v) for k, v in rouge1_scores.items()},
            rouge2={k: np.mean(v) for k, v in rouge2_scores.items()},
            rougeL={k: np.mean(v) for k, v in rougeL_scores.items()}
        )
        
        return rouge_scores
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for ROUGE calculation."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation except periods (for sentence boundaries)
        text = re.sub(r'[^\w\s.]', ' ', text)
        
        # Split into tokens
        tokens = text.split()
        
        return tokens
    
    def _calculate_rouge_n(self, pred_tokens: List[str], 
                          ref_tokens: List[str], n: int) -> Dict[str, float]:
        """Calculate ROUGE-N scores."""
        # Get n-grams
        pred_ngrams = self._get_ngrams(pred_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        
        # Count overlapping n-grams
        pred_counter = Counter(pred_ngrams)
        ref_counter = Counter(ref_ngrams)
        
        overlap = 0
        for ngram in pred_counter:
            overlap += min(pred_counter[ngram], ref_counter.get(ngram, 0))
        
        # Calculate metrics
        precision = overlap / len(pred_ngrams) if pred_ngrams else 0.0
        recall = overlap / len(ref_ngrams) if ref_ngrams else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Extract n-grams from tokens."""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return ngrams
    
    def _calculate_rouge_l(self, pred_tokens: List[str], 
                          ref_tokens: List[str]) -> Dict[str, float]:
        """Calculate ROUGE-L scores using longest common subsequence."""
        lcs_length = self._lcs_length(pred_tokens, ref_tokens)
        
        precision = lcs_length / len(pred_tokens) if pred_tokens else 0.0
        recall = lcs_length / len(ref_tokens) if ref_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n] 