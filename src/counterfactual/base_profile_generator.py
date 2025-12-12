"""
Base Profile Generator copied and adapted from Potemkin.
"""

from typing import List, Set, Dict, Tuple, Any, Optional
import random
import numpy as np
import polars as pl
from datasets import Dataset
from tqdm.auto import tqdm
import yaml

from .util import format_value

class BaseProfileGenerator:
    """
    A base class for generating descriptive profiles from tabular data based on feature, treatment, and outcome configurations.
    
    This class provides functionality to:
    - Validate input data against feature configurations
    - Generate descriptive statistics for features
    - Create template profiles from individual rows
    - Generate treatment profiles describing promotional campaign details
    - Generate outcome profiles describing campaign results
    - Convert tabular data into descriptive text profiles
    
    The class expects features to be organized by domains and supports various data types
    including numeric, binary, and string features. Treatment and outcome configurations
    are optional and enable generation of specialized profiles for promotional campaigns.
    """
    
    def __init__(self,
                 df: pl.DataFrame,
                 feature_config: str,
                 treatment_config: Optional[str] = None,
                 outcome_config: Optional[str] = None,
                 row_limit: Optional[int] = None,
                 shuffle_features: Optional[bool] = False, 
                 shuffle_domains: Optional[bool] = False,
                 feature_distribution_exclude_cols: Optional[List[str]] = None):
        """
        Initialize the BaseProfileGenerator.
        
        Args:
            df (pl.DataFrame): Input Polars DataFrame containing the features to generate profiles from
            feature_config (str): Path to feature configuration YAML file. Defaults to 'src/config/counterfactual/feature_config.yaml'
            treatment_config (str, optional): Path to treatment configuration YAML file
            outcome_config (str, optional): Path to outcome configuration YAML file
            row_limit (int, optional): Limit # rows for quick testing
            shuffle_features (bool, optional): Whether to randomize feature order within domains
            shuffle_domains (bool, optional): Whether to randomize domain order
            feature_distribution_exclude_cols (List[str], optional): List of columns to exclude from feature distribution calculation
        
        The initialization process:
        1. Stores the input DataFrame and configuration paths
        2. Loads all configurations from YAML files
        3. Validates DataFrame columns against feature config
        4. Applies row limit if specified
        5. Creates a template profile dataset
        """
        self.df = df
        

        # Load all configurations from YAML files
        self.feature_config = self._load_feature_config(feature_config)
        self.treatment_config = self._load_treatment_config(treatment_config) if treatment_config else None
        self.outcome_config = self._load_outcome_config(outcome_config) if outcome_config else None
        
        self.feature_distribution = self.describe_feature_distribution(feature_distribution_exclude_cols=feature_distribution_exclude_cols)
        self.validate_df_columns()
        if row_limit:
            self.df = self.df.sample(n=row_limit, seed=42)
        self.template_profile_dataset = self.generate_template_profile_dataset(shuffle_features, shuffle_domains)
    
    def _load_feature_config(self, feature_config_path: str) -> List[Dict[str, Any]]:
        """
        Load feature configuration from YAML file.
        
        Args:
            feature_config_path (str): Path to feature configuration YAML file
            
        Returns:
            List[Dict[str, Any]]: Feature configuration list
        """
        with open(feature_config_path, 'r') as f:
            customer_features = yaml.safe_load(f)
        
        feature_config = []
        for feature_name, mapping in customer_features['customer_features'].items():
            if feature_name in self.df.columns:
                feature_config.append({
                    'feature_name': feature_name,
                    'feature_description': mapping['description'],
                    'feature_domain': mapping['domain'],
                    'data_type': mapping['data_type']
                })
        
        return feature_config
    
    def _load_treatment_config(self, treatment_config_path: str) -> List[Dict[str, Any]]:
        """
        Load treatment configuration from YAML file.
        
        Args:
            treatment_config_path (str): Path to treatment configuration YAML file
            
        Returns:
            List[Dict[str, Any]]: Treatment configuration list
        """
        with open(treatment_config_path, 'r') as f:
            treatment_config = yaml.safe_load(f)
        
        return treatment_config.get('treatment_features', [])
    
    def _load_outcome_config(self, outcome_config_path: str) -> List[Dict[str, Any]]:
        """
        Load outcome configuration from YAML file.
        
        Args:
            outcome_config_path (str): Path to outcome configuration YAML file
            
        Returns:
            List[Dict[str, Any]]: Outcome configuration list
        """
        with open(outcome_config_path, 'r') as f:
            outcome_config = yaml.safe_load(f)
        
        return outcome_config.get('outcome_features', [])
    
    def validate_df_columns(self) -> None:
        """
        Validate that all features specified in feature_config exist in the DataFrame.
        
        This method ensures data integrity by checking that every feature mentioned
        in the feature configuration is present as a column in the input DataFrame.
        
        Raises:
            ValueError: If feature_config is not a list or if any feature is missing required fields
            ValueError: If any features from feature_config are missing in the dataframe
        """
        feature_names = [feature['feature_name'] for feature in self.feature_config]
        missing_features = [name for name in feature_names if name not in self.df.columns]
        if missing_features:
            raise ValueError(f"Missing features in dataframe: {missing_features}")

    def describe_feature_distribution(self, feature_distribution_exclude_cols: List[str] = None) -> str:
        """
        Generate descriptive statistics for features specified in feature_config
        Args:
            feature_distribution_exclude_cols (List[str]): List of columns to exclude
            
        Returns:
            str: Generated paragraphs describing the distribution of each feature.
            - For binary features: only mean is reported
            - For numeric features: min, max, mean, and deciles
            - For string features: normalized value counts
        """
        
        # Group features by domain
        domain_features = {}
        for feature in self.feature_config:
            domain = feature['feature_domain']
            if domain not in domain_features:
                domain_features[domain] = []
            domain_features[domain].append(feature)
        
        paragraphs = []
        
        # Generate statistics for each domain
        for domain, features in domain_features.items():
            domain_paragraphs = []
            
            for feature in features:
                #skip if excluded
                if feature['feature_name'] in feature_distribution_exclude_cols:
                    continue

                feature_name = feature['feature_name']
                if feature_name not in self.df.columns:
                    continue
                    
                row = self.df[feature_name]

                # Handle string features
                if feature['data_type'] == 'string':
                    value_counts = row.value_counts(sort=True, parallel=True)
                    total_count = len(row)
                    lines = [f"Distribution of {feature['feature_description']}:"]
                    lines.append("- Value counts:")
                    for item in value_counts.iter_rows():
                        value, count = item
                        percentage = count / total_count
                        lines.append(f"  {value}: {percentage:.2%}")
                    feature_text = "\n".join(lines)
                    domain_paragraphs.append(feature_text)
                
                # Handle binary features
                elif feature['data_type'] == 'binary':
                    mean = row.mean()
                    feature_text = (f"Distribution of {feature['feature_description']}:\n"
                                f"- Mean: {mean:.2%}")
                    domain_paragraphs.append(feature_text)

                # Calculate statistics for numeric features
                else:
                    stats = {
                        'min': row.min(),
                        'max': row.max(),
                        'mean': row.mean(),
                    }
                    
                    # Calculate percentiles using Polars (one at a time)
                    percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    for i, percentile in enumerate(percentiles):
                        p = [10, 20, 30, 40, 50, 60, 70, 80, 90][i]
                        stats[f'p{p}'] = row.quantile(percentile)
                    
                    # Format all values according to feature's data type
                    formatted_stats = {
                        k: format_value(v, feature['data_type']) 
                        for k, v in stats.items()
                    }
                    
                    # Create description paragraph
                    lines = [f"Distribution of {feature['feature_description']}:"]
                    lines.append(f"- Range: {formatted_stats['min']} to {formatted_stats['max']}")
                    if 'days_since_' in feature['feature_name']:
                        under_360_pct = (row < 360).mean()
                        lines.append(f"- Percent of values under 360 days: {under_360_pct:.2%}")
                    else:
                        non_zero_pct = (row > 0).mean()
                        lines.append(f"- Percent of non-zero values: {non_zero_pct:.2%}")
                    lines.append(f"- Mean: {formatted_stats['mean']}")
                    lines.append("- Percentiles:")
                    for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
                        lines.append(f"  {p}th: {formatted_stats[f'p{p}']}")
                    
                    feature_text = "\n".join(lines)
                    domain_paragraphs.append(feature_text)
            
            if domain_paragraphs:
                domain_text = f"[{domain} Domain]\n" + "\n".join(domain_paragraphs)
                paragraphs.append(domain_text)
        
        return "\n".join(paragraphs)

    def generate_template_profile(self,
                                row: Dict[str, Any], 
                                shuffle_features: bool = False, 
                                shuffle_domains: bool = False) -> Dict[str, str]:
        """
        Generate a descriptive profile from a single row of tabular data.
        
        This method converts a row of data into a human-readable text profile,
        organizing features by their domains. Features within domains and the
        domains themselves can optionally be shuffled to create variety in the
        output text.
        
        Args:
            row (Dict[str, Any]): Row of tabular data to generate text profile from
            shuffle_features (bool, optional): Whether to randomize the order of features
                within each domain. Defaults to False.
            shuffle_domains (bool, optional): Whether to randomize the order of domains
                in the output. Defaults to False.
            
        Returns:
            Dict[str, str]: A dictionary containing:
                - customer_id: The ID of the customer from the input row
                - template_profile: Generated text profile with features organized by domains,
                  where each domain's features are presented as sentences and domains are
                  separated by newlines
        """

        paragraphs = []
        
        # Group features by domain
        domain_features = {}
        for feature in self.feature_config:
            domain = feature['feature_domain']
            if domain not in domain_features:
                domain_features[domain] = []
            domain_features[domain].append(feature)
        
        # Convert to list of (domain, features) tuples for potential shuffling
        domain_items = list(domain_features.items())
        
        # Shuffle domains if requested
        if shuffle_domains:
            random.shuffle(domain_items)
        
        # Generate text for each domain
        for domain, features in domain_items:
            domain_sentences = []
            
            # Create copy of features list for potential shuffling
            domain_features_list = features.copy()
            
            # Shuffle features within domain if requested
            if shuffle_features:
                random.shuffle(domain_features_list)
            
            for feature in domain_features_list:
                feature_name = feature['feature_name']
                if feature_name in row:
                    value = row[feature_name]
                    formatted_value = format_value(value, feature['data_type'])
                    
                    # Create sentence based on feature description
                    sentence = f"{feature['feature_description']}: {formatted_value}"
                    domain_sentences.append(sentence)
            
            if domain_sentences:
                domain_text = f"[{domain} Domain] " + ". ".join(domain_sentences) + "."
                paragraphs.append(domain_text)
        
        # Combine all domain paragraphs with newlines
        return {
            "customer_id": row["customer_id"],
            "template_profile": "\n".join(paragraphs)
        }

    def generate_template_treatment(self,
                                row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a descriptive treatment profile from a single row of tabular data.
        
        This method converts treatment-related features into both a human-readable text profile
        and a list of raw treatment values, similar to generate_template_outcome().
        
        Args:
            row (Dict[str, Any]): Row of tabular data containing treatment features
            
        Returns:
            Dict[str, Any]: A dictionary containing:
                - treatment: List of raw treatment values in the order they appear in treatment_config
                - template_treatment: Generated text profile with treatment features presented
                  as sentences joined with spaces
        """
        treatment_sentences = []
        treatment_values = []
        for feature in self.treatment_config:
            feature_name = feature['feature_name']
            value = row[feature_name]
            formatted_value = format_value(value, feature['data_type'])
            sentence = f"{feature['feature_description']}: {formatted_value}."
            treatment_sentences.append(sentence)
            treatment_values.append(value)
        return {
            "treatment": treatment_values,
            "template_treatment": " ".join(treatment_sentences)
        }

    def generate_template_outcome(self,
                                row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a descriptive outcome profile from a single row of tabular data.
        
        This method converts outcome-related features into both a human-readable text profile
        and a list of raw outcome values. The profile describes campaign results such as
        purchase amounts, points earned, and completion status.
        
        Args:
            row (Dict[str, Any]): Row of tabular data containing outcome features
            
        Returns:
            Dict[str, Any]: A dictionary containing:
                - outcome: List of raw outcome values in the order they appear in outcome_config
                - template_outcome: Generated text profile with outcome features presented
                  as sentences joined with spaces
        """
        outcome_sentences = []
        outcome_values = []
        for feature in self.outcome_config:
            feature_name = feature['feature_name']
            value = row[feature_name]
            formatted_value = format_value(value, feature['data_type'])
            sentence = f"{feature['feature_description']}: {formatted_value}."
            outcome_sentences.append(sentence)
            outcome_values.append(value)
        return {
            "outcome": [str(i) for i in outcome_values],
            "template_outcome": " ".join(outcome_sentences)
        }

    def generate_template_profile_dataset(self, 
                                          shuffle_features: bool = False, 
                                          shuffle_domains: bool = False) -> Dataset:
        """
        Generate a dataset of template profiles for all rows in the DataFrame.
        
        This method iterates through each row in the input DataFrame and generates
        a template profile using the generate_template_profile method. The results
        are compiled into a Dataset object.
        
        Returns:
            Dataset: A Hugging Face Dataset containing template profiles for each row,
                    with 'customer_id' and 'template_profile' columns
        """
        template_profile_list = []
        
        # Use Polars-native row iteration with to_dicts()
        for row_dict in tqdm(self.df.to_dicts()):
            total_profile = {}
            customer_profile = self.generate_template_profile(row_dict, shuffle_features, shuffle_domains)
            total_profile = total_profile|customer_profile
            
            if self.treatment_config:
                treatment_profile = self.generate_template_treatment(row_dict)
                total_profile = total_profile|treatment_profile
            if self.outcome_config:
                outcome_profile = self.generate_template_outcome(row_dict)
                total_profile = total_profile|outcome_profile
            template_profile_list.append(total_profile)
        return Dataset.from_list(template_profile_list)
