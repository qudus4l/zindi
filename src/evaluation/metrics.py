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
    
    def evaluate_batch(self, predictions: List[str], references: List[str]) -> Dict[str, Any]:
        """Evaluate a batch of predictions against references.
        
        Args:
            predictions: List of predicted text responses
            references: List of reference text responses
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Calculate ROUGE scores
        rouge_scores = self.calculate_rouge_scores(predictions, references)
        
        # Calculate clinical relevance metrics
        clinical_metrics = self._calculate_clinical_metrics(predictions, references)
        
        # Combine all metrics
        results = {
            'rouge_scores': {
                'rouge1': rouge_scores.rouge1,
                'rouge2': rouge_scores.rouge2,
                'rougeL': rouge_scores.rougeL
            },
            'clinical_metrics': clinical_metrics,
            'summary': {
                'rouge1_f1': rouge_scores.rouge1['f1'],
                'rouge2_f1': rouge_scores.rouge2['f1'],
                'rougeL_f1': rouge_scores.rougeL['f1'],
                'clinical_relevance': clinical_metrics['clinical_relevance'],
                'avg_response_length': clinical_metrics['avg_response_length']
            }
        }
        
        return results
    
    def format_results(self, results: Dict[str, Any]) -> str:
        """Format evaluation results for display.
        
        Args:
            results: Results dictionary from evaluate_batch
            
        Returns:
            Formatted string representation of results
        """
        output = []
        output.append("EVALUATION RESULTS")
        output.append("-" * 50)
        
        # ROUGE scores
        output.append("\nROUGE Scores:")
        rouge_scores = results['rouge_scores']
        for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
            scores = rouge_scores[rouge_type]
            output.append(f"  {rouge_type.upper()}:")
            output.append(f"    Precision: {scores['precision']:.4f}")
            output.append(f"    Recall:    {scores['recall']:.4f}")
            output.append(f"    F1:        {scores['f1']:.4f}")
        
        # Clinical metrics
        output.append("\nClinical Metrics:")
        clinical = results['clinical_metrics']
        output.append(f"  Clinical Relevance: {clinical['clinical_relevance']:.4f}")
        output.append(f"  Medical Keywords:   {clinical['medical_keyword_coverage']:.4f}")
        output.append(f"  Avg Response Len:   {clinical['avg_response_length']:.1f} words")
        output.append(f"  Response Variance:  {clinical['response_length_variance']:.1f}")
        
        # Summary
        output.append("\nSummary:")
        summary = results['summary']
        output.append(f"  Primary Score (ROUGE-1 F1): {summary['rouge1_f1']:.4f}")
        output.append(f"  Clinical Relevance:          {summary['clinical_relevance']:.4f}")
        
        return "\n".join(output)
    
    def _calculate_clinical_metrics(self, predictions: List[str], 
                                  references: List[str]) -> Dict[str, float]:
        """Calculate clinical-specific metrics.
        
        Args:
            predictions: List of predicted responses
            references: List of reference responses
            
        Returns:
            Dictionary of clinical metrics
        """
        # Calculate medical keyword coverage
        pred_keyword_scores = []
        ref_keyword_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_keywords = self._count_medical_keywords(pred)
            ref_keywords = self._count_medical_keywords(ref)
            
            pred_keyword_scores.append(pred_keywords)
            ref_keyword_scores.append(ref_keywords)
        
        # Calculate response length statistics
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]
        
        # Clinical relevance score (combination of keyword coverage and length similarity)
        clinical_relevance_scores = []
        for i in range(len(predictions)):
            keyword_sim = min(pred_keyword_scores[i], ref_keyword_scores[i]) / max(ref_keyword_scores[i], 1)
            length_sim = 1 - abs(pred_lengths[i] - ref_lengths[i]) / max(ref_lengths[i], 1)
            clinical_relevance = (keyword_sim + length_sim) / 2
            clinical_relevance_scores.append(clinical_relevance)
        
        return {
            'clinical_relevance': np.mean(clinical_relevance_scores),
            'medical_keyword_coverage': np.mean(pred_keyword_scores) / max(np.mean(ref_keyword_scores), 1),
            'avg_response_length': np.mean(pred_lengths),
            'response_length_variance': np.var(pred_lengths),
            'length_similarity': 1 - np.mean([abs(p - r) / max(r, 1) for p, r in zip(pred_lengths, ref_lengths)])
        }
    
    def _count_medical_keywords(self, text: str) -> int:
        """Count medical keywords in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Number of medical keywords found
        """
        text_lower = text.lower()
        count = 0
        for keyword in self.medical_keywords:
            if keyword in text_lower:
                count += 1
        return count
        
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