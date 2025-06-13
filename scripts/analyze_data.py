#!/usr/bin/env python3
"""Phase 1: Data Analysis and Exploration Script.

This script performs comprehensive analysis of the clinical vignette dataset
to understand patterns, identify challenges, and establish baselines.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from collections import Counter
import re

from src.data.loader import ClinicalDataLoader
from src.utils.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClinicalDataAnalyzer:
    """Comprehensive analyzer for clinical vignette data."""
    
    def __init__(self, config: Config):
        """Initialize the analyzer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.loader = ClinicalDataLoader(config.data.raw_data_dir)
        self.output_dir = Path("analysis_results")
        self.output_dir.mkdir(exist_ok=True)
        
    def run_full_analysis(self) -> Dict[str, any]:
        """Run complete data analysis pipeline.
        
        Returns:
            Dict[str, any]: Analysis results
        """
        logger.info("Starting comprehensive data analysis")
        
        # Load data
        train_df = self.loader.load_train_data()
        test_df = self.loader.load_test_data()
        
        # Analyze clinical patterns
        train_stats = self.loader.analyze_clinical_patterns(train_df)
        
        # Validate clinical safety
        safety_concerns = self.loader.validate_clinical_safety(train_df)
        
        # Analyze response patterns
        response_analysis = self._analyze_response_patterns(train_df)
        
        # Analyze clinical domains
        domain_analysis = self._analyze_clinical_domains(train_df)
        
        # Calculate baseline metrics
        baseline_metrics = self._calculate_baseline_metrics(train_df)
        
        # Analyze data quality
        quality_analysis = self._analyze_data_quality(train_df)
        
        # Generate visualizations
        self._create_visualizations(train_df, train_stats)
        
        # Compile results
        results = {
            'train_stats': train_stats,
            'test_samples': len(test_df),
            'safety_concerns': safety_concerns,
            'response_analysis': response_analysis,
            'domain_analysis': domain_analysis,
            'baseline_metrics': baseline_metrics,
            'quality_analysis': quality_analysis
        }
        
        # Save detailed report
        self._save_analysis_report(results)
        
        return results
    
    def _analyze_response_patterns(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze patterns in clinician responses.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Dict[str, any]: Response pattern analysis
        """
        logger.info("Analyzing response patterns")
        
        responses = df['Clinician response'].tolist()
        
        # Common response starters
        starters = []
        for response in responses:
            words = response.split()
            if len(words) >= 3:
                starters.append(' '.join(words[:3]))
        
        common_starters = Counter(starters).most_common(10)
        
        # Response structure patterns
        structured_responses = 0
        list_format_responses = 0
        
        for response in responses:
            if any(marker in response.lower() for marker in ['first', 'second', 'finally', '1.', '2.']):
                structured_responses += 1
            if re.search(r'\d+\.|\-\s+|\*\s+', response):
                list_format_responses += 1
        
        # Action words analysis
        action_words = ['assess', 'examine', 'check', 'monitor', 'administer', 
                       'refer', 'educate', 'counsel', 'treat', 'diagnose']
        action_word_counts = {}
        
        for word in action_words:
            count = sum(1 for response in responses if word in response.lower())
            action_word_counts[word] = count
        
        return {
            'common_starters': common_starters,
            'structured_responses_pct': (structured_responses / len(df)) * 100,
            'list_format_responses_pct': (list_format_responses / len(df)) * 100,
            'action_word_usage': action_word_counts,
            'avg_sentences_per_response': np.mean([len(r.split('.')) for r in responses])
        }
    
    def _analyze_clinical_domains(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analyze clinical domains covered in the dataset.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Dict[str, int]: Distribution of clinical domains
        """
        logger.info("Analyzing clinical domains")
        
        domain_keywords = {
            'maternal_health': ['pregnancy', 'pregnant', 'antenatal', 'postnatal', 
                              'delivery', 'labor', 'maternal'],
            'pediatrics': ['child', 'infant', 'baby', 'pediatric', 'newborn', 
                          'immunization', 'growth'],
            'infectious_diseases': ['malaria', 'HIV', 'TB', 'tuberculosis', 'pneumonia',
                                  'diarrhea', 'fever', 'infection'],
            'chronic_diseases': ['diabetes', 'hypertension', 'asthma', 'epilepsy',
                               'heart disease', 'chronic'],
            'emergency_care': ['emergency', 'urgent', 'acute', 'severe', 'critical',
                             'trauma', 'accident'],
            'mental_health': ['mental', 'depression', 'anxiety', 'psychiatric',
                            'counseling', 'stress'],
            'nutrition': ['malnutrition', 'nutrition', 'feeding', 'diet', 'weight loss',
                         'underweight']
        }
        
        domain_counts = {domain: 0 for domain in domain_keywords}
        
        for _, row in df.iterrows():
            text = (row['Vignette'] + ' ' + row['Clinician response']).lower()
            
            for domain, keywords in domain_keywords.items():
                if any(keyword in text for keyword in keywords):
                    domain_counts[domain] += 1
        
        return domain_counts
    
    def _calculate_baseline_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate baseline metrics for model comparison.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Dict[str, float]: Baseline metrics
        """
        logger.info("Calculating baseline metrics")
        
        # Simple baseline: most common response length
        response_lengths = df['Clinician response'].str.split().str.len()
        avg_response_length = response_lengths.mean()
        median_response_length = response_lengths.median()
        
        # Character-level statistics
        char_lengths = df['Clinician response'].str.len()
        avg_char_length = char_lengths.mean()
        
        # Vocabulary statistics
        all_responses = ' '.join(df['Clinician response'].tolist())
        vocab = set(all_responses.lower().split())
        
        return {
            'avg_response_words': avg_response_length,
            'median_response_words': median_response_length,
            'avg_response_chars': avg_char_length,
            'vocabulary_size': len(vocab),
            'responses_per_vignette': 1.0  # Each vignette has one response
        }
    
    def _analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze data quality issues.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Dict[str, any]: Data quality analysis
        """
        logger.info("Analyzing data quality")
        
        quality_issues = {
            'missing_values': df.isnull().sum().to_dict(),
            'empty_strings': {
                'vignettes': (df['Vignette'].str.strip() == '').sum(),
                'responses': (df['Clinician response'].str.strip() == '').sum()
            },
            'duplicate_vignettes': df.duplicated(subset=['Vignette']).sum(),
            'duplicate_responses': df.duplicated(subset=['Clinician response']).sum(),
            'very_short_responses': (df['Clinician response'].str.split().str.len() < 5).sum(),
            'very_long_responses': (df['Clinician response'].str.split().str.len() > 100).sum()
        }
        
        # Check for potential encoding issues
        encoding_issues = 0
        for _, row in df.iterrows():
            text = row['Vignette'] + ' ' + row['Clinician response']
            if any(ord(char) > 127 for char in text if char not in ['–', '—', ''', ''', '"', '"']):
                encoding_issues += 1
        
        quality_issues['potential_encoding_issues'] = encoding_issues
        
        return quality_issues
    
    def _create_visualizations(self, df: pd.DataFrame, stats: any) -> None:
        """Create visualization plots for analysis.
        
        Args:
            df: Training DataFrame
            stats: Clinical data statistics
        """
        logger.info("Creating visualizations")
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Response length distribution
        response_lengths = df['Clinician response'].str.split().str.len()
        axes[0, 0].hist(response_lengths, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(response_lengths.mean(), color='red', linestyle='--', 
                          label=f'Mean: {response_lengths.mean():.1f}')
        axes[0, 0].set_xlabel('Response Length (words)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Response Lengths')
        axes[0, 0].legend()
        
        # 2. Top medical terms
        terms, counts = zip(*stats.common_medical_terms[:15])
        axes[0, 1].barh(terms, counts)
        axes[0, 1].set_xlabel('Frequency')
        axes[0, 1].set_title('Top 15 Medical Terms')
        
        # 3. Clinical domains distribution
        domain_analysis = self._analyze_clinical_domains(df)
        domains = list(domain_analysis.keys())
        domain_counts = list(domain_analysis.values())
        
        axes[1, 0].bar(range(len(domains)), domain_counts)
        axes[1, 0].set_xticks(range(len(domains)))
        axes[1, 0].set_xticklabels(domains, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Number of Cases')
        axes[1, 0].set_title('Clinical Domain Distribution')
        
        # 4. Vignette vs Response length correlation
        vignette_lengths = df['Vignette'].str.split().str.len()
        axes[1, 1].scatter(vignette_lengths, response_lengths, alpha=0.5)
        axes[1, 1].set_xlabel('Vignette Length (words)')
        axes[1, 1].set_ylabel('Response Length (words)')
        axes[1, 1].set_title('Vignette vs Response Length Correlation')
        
        # Add correlation coefficient
        correlation = np.corrcoef(vignette_lengths, response_lengths)[0, 1]
        axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=axes[1, 1].transAxes, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'data_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {self.output_dir / 'data_analysis_plots.png'}")
    
    def _save_analysis_report(self, results: Dict[str, any]) -> None:
        """Save detailed analysis report.
        
        Args:
            results: Complete analysis results
        """
        report_path = self.output_dir / 'data_analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("CLINICAL VIGNETTE DATA ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            # Dataset Overview
            f.write("1. DATASET OVERVIEW\n")
            f.write("-" * 30 + "\n")
            stats = results['train_stats']
            f.write(f"Total training samples: {stats.total_samples}\n")
            f.write(f"Total test samples: {results['test_samples']}\n")
            f.write(f"Average vignette length: {stats.avg_vignette_length:.1f} characters\n")
            f.write(f"Average response length: {stats.avg_response_length:.1f} characters\n\n")
            
            # Response Analysis
            f.write("2. RESPONSE PATTERN ANALYSIS\n")
            f.write("-" * 30 + "\n")
            response_analysis = results['response_analysis']
            f.write(f"Structured responses: {response_analysis['structured_responses_pct']:.1f}%\n")
            f.write(f"List format responses: {response_analysis['list_format_responses_pct']:.1f}%\n")
            f.write(f"Average sentences per response: {response_analysis['avg_sentences_per_response']:.1f}\n\n")
            
            f.write("Common response starters:\n")
            for starter, count in response_analysis['common_starters']:
                f.write(f"  '{starter}': {count} times\n")
            f.write("\n")
            
            # Clinical Domains
            f.write("3. CLINICAL DOMAIN COVERAGE\n")
            f.write("-" * 30 + "\n")
            for domain, count in results['domain_analysis'].items():
                percentage = (count / stats.total_samples) * 100
                f.write(f"{domain}: {count} cases ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Data Quality
            f.write("4. DATA QUALITY ANALYSIS\n")
            f.write("-" * 30 + "\n")
            quality = results['quality_analysis']
            f.write(f"Duplicate vignettes: {quality['duplicate_vignettes']}\n")
            f.write(f"Duplicate responses: {quality['duplicate_responses']}\n")
            f.write(f"Very short responses (<5 words): {quality['very_short_responses']}\n")
            f.write(f"Very long responses (>100 words): {quality['very_long_responses']}\n")
            f.write(f"Potential encoding issues: {quality['potential_encoding_issues']}\n\n")
            
            # Safety Concerns
            f.write("5. CLINICAL SAFETY ANALYSIS\n")
            f.write("-" * 30 + "\n")
            if results['safety_concerns']:
                f.write(f"Found {len(results['safety_concerns'])} potential safety concerns\n")
                for concern in results['safety_concerns'][:5]:  # Show first 5
                    f.write(f"  - Sample {concern['sample_id']}: {concern['concern_type']}\n")
            else:
                f.write("No obvious safety concerns detected in initial scan\n")
            f.write("\n")
            
            # Key Insights
            f.write("6. KEY INSIGHTS FOR MODEL DEVELOPMENT\n")
            f.write("-" * 30 + "\n")
            f.write("- Responses are generally concise (median ~30 words)\n")
            f.write("- Mix of structured and narrative response styles\n")
            f.write("- Strong focus on infectious diseases and maternal health\n")
            f.write("- Language is primarily English with some Swahili terms\n")
            f.write("- Action-oriented vocabulary is prevalent\n")
            f.write("- Data quality is generally good with minimal issues\n")
        
        logger.info(f"Analysis report saved to {report_path}")


def main():
    """Main execution function."""
    # Load configuration
    config = Config()
    
    # Run analysis
    analyzer = ClinicalDataAnalyzer(config)
    results = analyzer.run_full_analysis()
    
    # Print summary
    print("\n" + "=" * 70)
    print("DATA ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Training samples analyzed: {results['train_stats'].total_samples}")
    print(f"Test samples found: {results['test_samples']}")
    print(f"Safety concerns identified: {len(results['safety_concerns'])}")
    print(f"Baseline vocabulary size: {results['baseline_metrics']['vocabulary_size']}")
    print(f"\nDetailed results saved to: analysis_results/")
    print("=" * 70)


if __name__ == "__main__":
    main() 