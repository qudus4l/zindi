#!/usr/bin/env python3
"""Phase 2: Model Architecture Selection and Evaluation.

This script evaluates different model architectures for the clinical decision
support system, focusing on parameter count, inference speed, and suitability
for edge deployment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates different model architectures for clinical decision support."""
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.output_dir = Path("model_evaluation_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Model candidates with their specifications
        self.model_candidates = {
            "t5-small": {
                "params": 60_000_000,  # 60M parameters
                "architecture": "encoder-decoder",
                "medical_pretrained": False,
                "quantization_friendly": True,
                "context_length": 512,
                "pros": [
                    "Excellent for text generation tasks",
                    "Good balance of size and performance",
                    "Well-suited for summarization and Q&A",
                    "Strong community support"
                ],
                "cons": [
                    "Not specifically trained on medical data",
                    "May need extensive fine-tuning"
                ]
            },
            "distilbert-base": {
                "params": 66_000_000,  # 66M parameters
                "architecture": "encoder-only",
                "medical_pretrained": False,
                "quantization_friendly": True,
                "context_length": 512,
                "pros": [
                    "Fast inference",
                    "Good for classification tasks",
                    "Distilled from BERT with minimal performance loss"
                ],
                "cons": [
                    "Encoder-only, needs additional decoder for generation",
                    "Limited generation capabilities"
                ]
            },
            "biobert-base": {
                "params": 110_000_000,  # 110M parameters
                "architecture": "encoder-only",
                "medical_pretrained": True,
                "quantization_friendly": True,
                "context_length": 512,
                "pros": [
                    "Pre-trained on biomedical literature",
                    "Better medical term understanding",
                    "Good for medical NER and classification"
                ],
                "cons": [
                    "Larger than other options",
                    "Encoder-only architecture",
                    "May need custom generation head"
                ]
            },
            "gpt2-small": {
                "params": 124_000_000,  # 124M parameters
                "architecture": "decoder-only",
                "medical_pretrained": False,
                "quantization_friendly": True,
                "context_length": 1024,
                "pros": [
                    "Strong text generation capabilities",
                    "Autoregressive generation",
                    "Good for conversational responses"
                ],
                "cons": [
                    "No medical pre-training",
                    "May generate less structured responses"
                ]
            },
            "clinical-t5-small": {
                "params": 60_000_000,  # 60M parameters (hypothetical)
                "architecture": "encoder-decoder",
                "medical_pretrained": True,
                "quantization_friendly": True,
                "context_length": 512,
                "pros": [
                    "Medical domain pre-training",
                    "T5 architecture benefits",
                    "Good for clinical text generation"
                ],
                "cons": [
                    "May not exist publicly",
                    "Would need custom training"
                ]
            },
            "custom-lightweight": {
                "params": 50_000_000,  # 50M parameters (target)
                "architecture": "encoder-decoder",
                "medical_pretrained": False,
                "quantization_friendly": True,
                "context_length": 512,
                "pros": [
                    "Designed specifically for this task",
                    "Optimized for edge deployment",
                    "Can incorporate clinical reasoning layers"
                ],
                "cons": [
                    "Requires development from scratch",
                    "No pre-training benefits",
                    "Higher development time"
                ]
            }
        }
        
    def evaluate_models(self) -> Dict[str, Any]:
        """Evaluate all model candidates."""
        logger.info("Starting model architecture evaluation")
        
        evaluation_results = {}
        
        for model_name, specs in self.model_candidates.items():
            logger.info(f"Evaluating {model_name}")
            
            # Calculate scores
            scores = self._calculate_model_scores(model_name, specs)
            
            # Estimate performance metrics
            performance = self._estimate_performance_metrics(specs)
            
            # Assess clinical suitability
            clinical_assessment = self._assess_clinical_suitability(model_name, specs)
            
            evaluation_results[model_name] = {
                "specifications": specs,
                "scores": scores,
                "performance_estimates": performance,
                "clinical_assessment": clinical_assessment,
                "overall_score": scores["overall"]
            }
        
        # Rank models
        ranked_models = self._rank_models(evaluation_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(ranked_models)
        
        # Save results
        self._save_evaluation_results(evaluation_results, ranked_models, recommendations)
        
        return {
            "evaluation_results": evaluation_results,
            "ranked_models": ranked_models,
            "recommendations": recommendations
        }
    
    def _calculate_model_scores(self, model_name: str, specs: Dict[str, Any]) -> Dict[str, float]:
        """Calculate various scores for a model."""
        scores = {}
        
        # Size score (smaller is better, max 1B params)
        size_ratio = specs["params"] / 1_000_000_000
        scores["size"] = max(0, 1 - size_ratio) * 100
        
        # Medical relevance score
        scores["medical_relevance"] = 100 if specs["medical_pretrained"] else 50
        
        # Architecture suitability for generation
        arch_scores = {
            "encoder-decoder": 100,
            "decoder-only": 80,
            "encoder-only": 40
        }
        scores["architecture"] = arch_scores.get(specs["architecture"], 50)
        
        # Quantization friendliness
        scores["quantization"] = 100 if specs["quantization_friendly"] else 50
        
        # Context length adequacy (based on avg vignette length ~100 words)
        scores["context"] = 100 if specs["context_length"] >= 512 else 70
        
        # Overall weighted score
        weights = {
            "size": 0.25,
            "medical_relevance": 0.30,
            "architecture": 0.20,
            "quantization": 0.15,
            "context": 0.10
        }
        
        scores["overall"] = sum(scores[key] * weights[key] for key in weights)
        
        return scores
    
    def _estimate_performance_metrics(self, specs: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate performance metrics for a model."""
        # These are rough estimates based on model size and architecture
        
        # Inference time estimation (ms per token)
        base_time = specs["params"] / 1_000_000  # Rough estimate
        
        # Architecture multipliers
        arch_multipliers = {
            "encoder-decoder": 1.5,  # Two-stage processing
            "decoder-only": 1.0,
            "encoder-only": 0.7  # Faster but needs generation head
        }
        
        ms_per_token = base_time * arch_multipliers.get(specs["architecture"], 1.0)
        
        # For 100-word response (~150 tokens)
        inference_time_ms = ms_per_token * 150
        
        # Memory estimation (MB)
        # Rough: 4 bytes per parameter + overhead
        memory_mb = (specs["params"] * 4 / 1_000_000) * 1.5  # 1.5x for overhead
        
        # Quantization benefits
        if specs["quantization_friendly"]:
            quantized_inference_time = inference_time_ms * 0.6
            quantized_memory = memory_mb * 0.25
        else:
            quantized_inference_time = inference_time_ms
            quantized_memory = memory_mb
        
        return {
            "inference_time_ms": round(inference_time_ms, 1),
            "memory_mb": round(memory_mb, 1),
            "quantized_inference_time_ms": round(quantized_inference_time, 1),
            "quantized_memory_mb": round(quantized_memory, 1),
            "meets_constraints": (
                quantized_inference_time < 100 and 
                quantized_memory < 2000
            )
        }
    
    def _assess_clinical_suitability(self, model_name: str, specs: Dict[str, Any]) -> Dict[str, Any]:
        """Assess clinical suitability of a model."""
        assessment = {
            "medical_knowledge": "High" if specs["medical_pretrained"] else "Low",
            "structured_output": "High" if specs["architecture"] == "encoder-decoder" else "Medium",
            "reasoning_capability": "Medium",  # All models need fine-tuning
            "safety_features": "Requires implementation",
            "deployment_readiness": "High" if specs["params"] < 100_000_000 else "Medium"
        }
        
        # Special cases
        if "clinical" in model_name.lower() or "bio" in model_name.lower():
            assessment["medical_knowledge"] = "High"
            assessment["reasoning_capability"] = "Medium-High"
        
        if model_name == "custom-lightweight":
            assessment["structured_output"] = "High"
            assessment["safety_features"] = "Can be built-in"
            assessment["deployment_readiness"] = "Low (requires development)"
        
        return assessment
    
    def _rank_models(self, evaluation_results: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Rank models by overall score."""
        ranked = []
        for model_name, results in evaluation_results.items():
            ranked.append((model_name, results["overall_score"]))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
    
    def _generate_recommendations(self, ranked_models: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Generate recommendations based on evaluation."""
        top_model = ranked_models[0][0]
        
        recommendations = {
            "primary_recommendation": top_model,
            "alternative_options": [model for model, _ in ranked_models[1:3]],
            "implementation_strategy": self._get_implementation_strategy(top_model),
            "risk_mitigation": self._get_risk_mitigation(top_model),
            "optimization_plan": self._get_optimization_plan(top_model)
        }
        
        return recommendations
    
    def _get_implementation_strategy(self, model_name: str) -> List[str]:
        """Get implementation strategy for a model."""
        strategies = {
            "t5-small": [
                "Start with base T5-small model",
                "Fine-tune on medical literature first",
                "Then fine-tune on Kenyan clinical data",
                "Implement clinical safety layers",
                "Apply quantization for edge deployment"
            ],
            "clinical-t5-small": [
                "Source or create medical T5 variant",
                "Fine-tune on Kenyan clinical data",
                "Implement structured output formatting",
                "Add clinical reasoning chains",
                "Optimize for edge deployment"
            ],
            "custom-lightweight": [
                "Design architecture with clinical constraints",
                "Implement attention mechanisms for reasoning",
                "Train from scratch with data augmentation",
                "Build in safety features from start",
                "Optimize architecture for quantization"
            ]
        }
        
        return strategies.get(model_name, ["Develop custom implementation plan"])
    
    def _get_risk_mitigation(self, model_name: str) -> List[str]:
        """Get risk mitigation strategies."""
        return [
            "Implement clinical validation at every step",
            "Use ensemble methods for critical decisions",
            "Add confidence scoring to outputs",
            "Create fallback mechanisms for edge cases",
            "Continuous monitoring of model outputs"
        ]
    
    def _get_optimization_plan(self, model_name: str) -> List[str]:
        """Get optimization plan for edge deployment."""
        return [
            "Apply INT8 quantization as baseline",
            "Test INT4 quantization impact on accuracy",
            "Implement dynamic quantization for layers",
            "Use ONNX Runtime for inference",
            "Profile and optimize bottlenecks",
            "Consider model pruning if needed"
        ]
    
    def _save_evaluation_results(self, evaluation_results: Dict[str, Any], 
                               ranked_models: List[Tuple[str, float]], 
                               recommendations: Dict[str, Any]) -> None:
        """Save evaluation results to files."""
        # Save detailed results as JSON
        results_path = self.output_dir / "model_evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                "evaluation_results": evaluation_results,
                "ranked_models": ranked_models,
                "recommendations": recommendations
            }, f, indent=2)
        
        # Save summary report
        report_path = self.output_dir / "model_evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write("MODEL ARCHITECTURE EVALUATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("1. MODEL RANKINGS\n")
            f.write("-" * 30 + "\n")
            for i, (model, score) in enumerate(ranked_models, 1):
                f.write(f"{i}. {model}: {score:.1f}/100\n")
            f.write("\n")
            
            f.write("2. TOP MODEL ANALYSIS\n")
            f.write("-" * 30 + "\n")
            top_model = ranked_models[0][0]
            top_results = evaluation_results[top_model]
            
            f.write(f"Model: {top_model}\n")
            f.write(f"Parameters: {top_results['specifications']['params']:,}\n")
            f.write(f"Architecture: {top_results['specifications']['architecture']}\n")
            f.write(f"Medical Pre-training: {top_results['specifications']['medical_pretrained']}\n")
            f.write(f"Estimated Inference Time: {top_results['performance_estimates']['quantized_inference_time_ms']}ms\n")
            f.write(f"Estimated Memory: {top_results['performance_estimates']['quantized_memory_mb']}MB\n")
            f.write("\n")
            
            f.write("3. RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Primary: {recommendations['primary_recommendation']}\n")
            f.write(f"Alternatives: {', '.join(recommendations['alternative_options'])}\n")
            f.write("\n")
            
            f.write("Implementation Strategy:\n")
            for step in recommendations['implementation_strategy']:
                f.write(f"  - {step}\n")
            
        logger.info(f"Evaluation results saved to {self.output_dir}")


def main():
    """Main execution function."""
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_models()
    
    print("\n" + "=" * 70)
    print("MODEL EVALUATION COMPLETE")
    print("=" * 70)
    
    print("\nTop 3 Models:")
    for i, (model, score) in enumerate(results['ranked_models'][:3], 1):
        print(f"{i}. {model}: {score:.1f}/100")
    
    print(f"\nRecommended Model: {results['recommendations']['primary_recommendation']}")
    print(f"\nDetailed results saved to: model_evaluation_results/")
    print("=" * 70)


if __name__ == "__main__":
    main() 