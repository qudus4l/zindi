"""Data loader for clinical vignettes with safety validation.

This module handles loading and initial validation of clinical data,
ensuring data integrity and identifying potential safety concerns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import re
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class ClinicalDataStats:
    """Statistics about the clinical dataset.
    
    Attributes:
        total_samples: Total number of samples
        avg_vignette_length: Average vignette text length
        avg_response_length: Average clinician response length
        response_length_distribution: Distribution of response lengths
        nurse_experience_distribution: Distribution of nurse experience levels
        facility_type_distribution: Distribution of facility types
        common_medical_terms: Most common medical terms
        language_patterns: Language usage patterns (English/Swahili)
    """
    total_samples: int
    avg_vignette_length: float
    avg_response_length: float
    response_length_distribution: Dict[str, int]
    nurse_experience_distribution: Dict[str, int]
    facility_type_distribution: Dict[str, int]
    common_medical_terms: List[Tuple[str, int]]
    language_patterns: Dict[str, Any]


class ClinicalDataLoader:
    """Loader for clinical vignette data with comprehensive validation.
    
    This class handles loading clinical data while performing safety checks
    and gathering statistics for model development.
    """
    
    def __init__(self, data_dir: Path):
        """Initialize the data loader.
        
        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = Path(data_dir)
        self.medical_terms_pattern = re.compile(
            r'\b(patient|diagnosis|treatment|symptom|medication|fever|pain|'
            r'cough|vomit|diarrhea|malaria|pneumonia|HIV|TB|diabetes|'
            r'hypertension|pregnancy|infant|child|adult|emergency)\b',
            re.IGNORECASE
        )
        self.swahili_pattern = re.compile(
            r'\b(mgonjwa|dawa|homa|maumivu|kikohozi|kutapika|kuhara|'
            r'malaria|nimonia|ukimwi|kifua kikuu|kisukari|shinikizo|'
            r'mjamzito|mtoto|mtu mzima)\b',
            re.IGNORECASE
        )
        
    def load_train_data(self, filename: str = "train.csv") -> pd.DataFrame:
        """Load training data with validation.
        
        Args:
            filename: Name of the training file
            
        Returns:
            pd.DataFrame: Loaded and validated training data
            
        Raises:
            FileNotFoundError: If data file not found
            ValueError: If data validation fails
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Training data not found at {filepath}")
        
        logger.info(f"Loading training data from {filepath}")
        
        # Load data
        df = pd.read_csv(filepath)
        
        # Validate required columns - using actual column names from the CSV
        required_columns = ['Master_Index', 'Prompt', 'Clinician']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            # Try alternative column names
            alt_columns = ['ID_VBWWP', 'Vignette', 'Clinician response']
            missing_alt = set(alt_columns) - set(df.columns)
            if len(missing_alt) < len(missing_columns):
                # Use alternative names
                df = df.rename(columns={'ID_VBWWP': 'Master_Index', 
                                      'Vignette': 'Prompt',
                                      'Clinician response': 'Clinician'})
            else:
                raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for missing values
        missing_counts = df[['Master_Index', 'Prompt', 'Clinician']].isnull().sum()
        if missing_counts.any():
            logger.warning(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
        
        # Validate text fields
        empty_vignettes = df[df['Prompt'].str.strip() == ''].shape[0]
        empty_responses = df[df['Clinician'].str.strip() == ''].shape[0]
        
        if empty_vignettes > 0:
            logger.warning(f"Found {empty_vignettes} empty vignettes")
        if empty_responses > 0:
            logger.warning(f"Found {empty_responses} empty responses")
        
        logger.info(f"Loaded {len(df)} training samples")
        
        return df
    
    def load_test_data(self, filename: str = "test.csv") -> pd.DataFrame:
        """Load test data for prediction.
        
        Args:
            filename: Name of the test file
            
        Returns:
            pd.DataFrame: Loaded test data
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Test data not found at {filepath}")
        
        logger.info(f"Loading test data from {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Validate required columns - handle different column names
        if 'Master_Index' not in df.columns:
            if 'ID' in df.columns:
                df = df.rename(columns={'ID': 'Master_Index'})
            else:
                raise ValueError("Test data must contain 'Master_Index' or 'ID' column")
        
        if 'Prompt' not in df.columns:
            if 'Vignette' in df.columns:
                df = df.rename(columns={'Vignette': 'Prompt'})
            else:
                raise ValueError("Test data must contain 'Prompt' or 'Vignette' column")
        
        logger.info(f"Loaded {len(df)} test samples")
        
        return df
    
    def analyze_clinical_patterns(self, df: pd.DataFrame) -> ClinicalDataStats:
        """Analyze clinical patterns in the dataset.
        
        Args:
            df: DataFrame with clinical data
            
        Returns:
            ClinicalDataStats: Comprehensive statistics about the dataset
        """
        logger.info("Analyzing clinical patterns in dataset")
        
        # Basic statistics
        total_samples = len(df)
        avg_vignette_length = df['Prompt'].str.len().mean()
        avg_response_length = df['Clinician'].str.len().mean()
        
        # Response length distribution
        response_lengths = df['Clinician'].str.split().str.len()
        response_length_bins = pd.cut(response_lengths, 
                                     bins=[0, 10, 20, 30, 50, 100, float('inf')],
                                     labels=['<10', '10-20', '20-30', '30-50', '50-100', '>100'])
        response_length_distribution = response_length_bins.value_counts().to_dict()
        
        # Extract nurse experience if available
        nurse_experience_distribution = self._extract_experience_levels(df)
        
        # Extract facility types if available
        facility_type_distribution = self._extract_facility_types(df)
        
        # Extract common medical terms
        all_text = ' '.join(df['Prompt'].tolist() + df['Clinician'].tolist())
        medical_terms = self.medical_terms_pattern.findall(all_text.lower())
        common_medical_terms = Counter(medical_terms).most_common(20)
        
        # Language patterns
        language_patterns = self._analyze_language_patterns(df)
        
        stats = ClinicalDataStats(
            total_samples=total_samples,
            avg_vignette_length=avg_vignette_length,
            avg_response_length=avg_response_length,
            response_length_distribution=response_length_distribution,
            nurse_experience_distribution=nurse_experience_distribution,
            facility_type_distribution=facility_type_distribution,
            common_medical_terms=common_medical_terms,
            language_patterns=language_patterns
        )
        
        self._log_statistics(stats)
        
        return stats
    
    def _extract_experience_levels(self, df: pd.DataFrame) -> Dict[str, int]:
        """Extract nurse experience levels from vignettes.
        
        Args:
            df: DataFrame with clinical data
            
        Returns:
            Dict[str, int]: Distribution of experience levels
        """
        experience_pattern = re.compile(
            r'(\d+)\s*years?\s*(?:of\s*)?(?:nursing\s*)?experience',
            re.IGNORECASE
        )
        
        experience_levels = []
        for vignette in df['Prompt']:
            match = experience_pattern.search(vignette)
            if match:
                years = int(match.group(1))
                if years < 2:
                    experience_levels.append('0-2 years')
                elif years < 5:
                    experience_levels.append('2-5 years')
                elif years < 10:
                    experience_levels.append('5-10 years')
                else:
                    experience_levels.append('10+ years')
            else:
                experience_levels.append('Unknown')
        
        return Counter(experience_levels)
    
    def _extract_facility_types(self, df: pd.DataFrame) -> Dict[str, int]:
        """Extract healthcare facility types from vignettes.
        
        Args:
            df: DataFrame with clinical data
            
        Returns:
            Dict[str, int]: Distribution of facility types
        """
        facility_keywords = {
            'hospital': ['hospital', 'referral'],
            'health_center': ['health center', 'health centre', 'HC'],
            'dispensary': ['dispensary'],
            'clinic': ['clinic'],
            'community': ['community', 'village']
        }
        
        facility_types = []
        for vignette in df['Prompt']:
            vignette_lower = vignette.lower()
            facility_found = False
            
            for facility_type, keywords in facility_keywords.items():
                if any(keyword in vignette_lower for keyword in keywords):
                    facility_types.append(facility_type)
                    facility_found = True
                    break
            
            if not facility_found:
                facility_types.append('Unknown')
        
        return Counter(facility_types)
    
    def _analyze_language_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze language usage patterns (English/Swahili).
        
        Args:
            df: DataFrame with clinical data
            
        Returns:
            Dict[str, Any]: Language pattern analysis
        """
        english_count = 0
        swahili_count = 0
        mixed_count = 0
        
        for _, row in df.iterrows():
            text = row['Prompt'] + ' ' + row['Clinician']
            
            english_matches = len(self.medical_terms_pattern.findall(text))
            swahili_matches = len(self.swahili_pattern.findall(text))
            
            if english_matches > 0 and swahili_matches > 0:
                mixed_count += 1
            elif swahili_matches > english_matches:
                swahili_count += 1
            else:
                english_count += 1
        
        return {
            'primarily_english': english_count,
            'primarily_swahili': swahili_count,
            'mixed_language': mixed_count,
            'percentage_mixed': (mixed_count / len(df)) * 100
        }
    
    def _log_statistics(self, stats: ClinicalDataStats) -> None:
        """Log dataset statistics for review.
        
        Args:
            stats: Clinical data statistics
        """
        logger.info("=" * 50)
        logger.info("CLINICAL DATASET ANALYSIS")
        logger.info("=" * 50)
        logger.info(f"Total samples: {stats.total_samples}")
        logger.info(f"Average vignette length: {stats.avg_vignette_length:.1f} characters")
        logger.info(f"Average response length: {stats.avg_response_length:.1f} characters")
        
        logger.info("\nResponse Length Distribution:")
        for length_range, count in sorted(stats.response_length_distribution.items()):
            logger.info(f"  {length_range} words: {count} samples")
        
        logger.info("\nNurse Experience Distribution:")
        for experience, count in stats.nurse_experience_distribution.items():
            logger.info(f"  {experience}: {count} samples")
        
        logger.info("\nFacility Type Distribution:")
        for facility, count in stats.facility_type_distribution.items():
            logger.info(f"  {facility}: {count} samples")
        
        logger.info("\nTop Medical Terms:")
        for term, count in stats.common_medical_terms[:10]:
            logger.info(f"  {term}: {count} occurrences")
        
        logger.info("\nLanguage Patterns:")
        for pattern, value in stats.language_patterns.items():
            logger.info(f"  {pattern}: {value}")
        
        logger.info("=" * 50)
    
    def validate_clinical_safety(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate dataset for potential clinical safety issues.
        
        Args:
            df: DataFrame with clinical data
            
        Returns:
            List[Dict[str, Any]]: List of potential safety concerns
        """
        safety_concerns = []
        
        # Check for potentially harmful advice patterns
        harmful_patterns = [
            (r'stop\s+all\s+medication', 'Advising to stop all medication'),
            (r'ignore\s+symptoms', 'Advising to ignore symptoms'),
            (r'no\s+need\s+(?:for\s+)?(?:to\s+)?(?:see|visit)\s+doctor', 'Discouraging medical consultation'),
            (r'traditional\s+medicine\s+only', 'Recommending only traditional medicine'),
        ]
        
        for idx, row in df.iterrows():
            response = row['Clinician'].lower()
            
            for pattern, description in harmful_patterns:
                if re.search(pattern, response):
                    safety_concerns.append({
                        'sample_id': row.get('Master_Index', row.get('ID', idx)),
                        'concern_type': description,
                        'pattern_found': pattern,
                        'response_snippet': response[:100] + '...'
                    })
        
        if safety_concerns:
            logger.warning(f"Found {len(safety_concerns)} potential safety concerns in responses")
        else:
            logger.info("No obvious safety concerns detected in initial scan")
        
        return safety_concerns
    
    def create_train_val_split(self, df: pd.DataFrame, val_ratio: float = 0.2, 
                             random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create training and validation splits with stratification.
        
        Args:
            df: Full training DataFrame
            val_ratio: Ratio of data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and validation DataFrames
        """
        # Shuffle data
        df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Calculate split point
        val_size = int(len(df_shuffled) * val_ratio)
        
        # Split data
        val_df = df_shuffled[:val_size]
        train_df = df_shuffled[val_size:]
        
        logger.info(f"Created train/val split: {len(train_df)} train, {len(val_df)} validation samples")
        
        return train_df, val_df 