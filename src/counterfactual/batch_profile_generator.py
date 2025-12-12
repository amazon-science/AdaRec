"""
Batch Profile Generator for counterfactual analysis.

Extends the base profile generator to support batch processing of multiple
customers per API call for efficient counterfactual generation.
"""

from typing import List, Dict, Any, Optional
import yaml
import polars as pl
import dspy
from datasets import Dataset
import time
import boto3
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

from .base_profile_generator import BaseProfileGenerator


class BatchProfileGenerator(BaseProfileGenerator):
    """
    Batch Profile Generator that extends BaseProfileGenerator for efficient
    counterfactual generation with multiple customers per API call.
    
    This class processes customer data in batches to generate narrative profiles
    suitable for counterfactual analysis of BxGy promotions.
    """
    
    def __init__(
        self,
        df: pl.DataFrame,
        feature_config: str,
        model_config: str,
        treatment_config: Optional[str] = None,
        outcome_config: Optional[str] = None,
        row_limit: Optional[int] = None,
        shuffle_features: Optional[bool] = False,
        shuffle_domains: Optional[bool] = False,
        feature_distribution_exclude_cols: Optional[List[str]] = None
    ):
        """
        Initialize the BatchProfileGenerator.
        
        Args:
            df (pl.DataFrame): Input Polars DataFrame with customer features
            feature_config (str): Path to feature configuration YAML file
            model_config (str): Path to model configuration YAML file
            treatment_config (str, optional): Path to treatment configuration YAML file
            outcome_config (str, optional): Path to outcome configuration YAML file
            row_limit (int, optional): Limit number of rows for testing
            shuffle_features (bool, optional): Whether to randomize feature order within domains
            shuffle_domains (bool, optional): Whether to randomize domain order
            feature_distribution_exclude_cols (List[str], optional): List of columns to exclude from feature distribution calculation
        """
        self.df = df
        self.row_limit = row_limit
        self.shuffle_features = shuffle_features
        self.shuffle_domains = shuffle_domains
        
        # Load model configuration
        with open(model_config, 'r') as f: self.config =  yaml.safe_load(f)
        
        # Initialize AWS Bedrock client
        try:
            self.bedrock = boto3.client('bedrock-runtime', region_name=self.config['bedrock']['region'])
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize AWS Bedrock client. "
                "Ensure AWS credentials are properly configured."
            ) from e
        
        # Initialize base class 
        super().__init__(
            df=df,
            feature_config=feature_config,
            treatment_config=treatment_config,
            outcome_config=outcome_config,
            row_limit=row_limit,
            shuffle_features=shuffle_features,
            shuffle_domains=shuffle_domains,
            feature_distribution_exclude_cols=feature_distribution_exclude_cols
        )
        
        # Configure DSPy with Bedrock
        profile_model_config = self.config['bedrock']['profile_model']
        dspy.settings.configure(
            lm=dspy.LM(
                model=profile_model_config['model_id'],
                temperature=profile_model_config['temperature'],
                max_tokens=profile_model_config['max_tokens'],
                num_retries=profile_model_config['num_retries']
            )
        )
        
        # Initialize DSPy predictor
        self.BatchNarrativeProfile.__doc__ = self.config['prompts']['profile_generation']['system_prompt']
        self.predictor = dspy.Predict(self.BatchNarrativeProfile)
        
        # Generate batch narrative profiles
        self.narrative_profile_dataset = None


    class BatchNarrativeProfile(dspy.Signature):
        """Generate narrative profiles for a batch of customers for counterfactual analysis."""
        feature_distribution: str = dspy.InputField(desc='Feature distribution based on all customers')
        customer_profiles_batch: str = dspy.InputField(desc='Batch of template customer profiles, one per line with format "Customer [ID]: [profile]"')
        batch_narrative_profiles: str = dspy.OutputField(desc='CRITICAL: Output a valid JSON array of customer profiles. Each object must have exactly two fields: "customer_id" (string) and "narrative_profile" (string). Example format: [{"customer_id": "123", "narrative_profile": "A detailed narrative..."}, {"customer_id": "456", "narrative_profile": "Another narrative..."}]. You MUST include ALL customers from the input in the same order. The output must be valid JSON that can be parsed with json.loads().')

    def generate_batch_narrative_profiles(
        self,
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None,
        seconds_per_batch: Optional[int] = None,
        chunk_size: Optional[int] = None,
        append_template: bool = False,
        debug_prompts: bool = False
    ) -> Dataset:
        """
        Generate narrative profiles for all customers in batches with true parallel processing.
        
        Args:
            batch_size (int, optional): Number of customers per batch window. Uses config default if None.
            max_workers (int, optional): Number of concurrent workers. Uses config default if None.
            seconds_per_batch (int, optional): Minimum seconds between batch windows. Uses config default if None.
            chunk_size (int, optional): Size of parallel chunks within each batch. Defaults to batch_size/max_workers.
            append_template (bool): If True, append the original template profile to the narrative profile.
                                   This ensures no information is lost during narrative generation.
            debug_prompts (bool): If True, print the actual prompts being sent to the LLM for debugging.
        
        Returns:
            Dataset: Dataset containing narrative profiles for all customers
        """
        # Use config defaults if not specified
        batch_size = batch_size or self.config['batch_processing']['profile_batch_size']
        max_workers = max_workers or self.config['batch_processing']['profile_max_workers']
        seconds_per_batch = seconds_per_batch or self.config['batch_processing']['profile_seconds_per_batch']
        
        # Calculate optimal chunk size for parallel processing
        if chunk_size is None:
            # Default: split batch into chunks that can be processed in parallel
            chunk_size = max(1, batch_size // max_workers)
        
        narrative_profiles = []
        nb_template_profiles = len(self.template_profile_dataset)
        
        print(f"Generating narrative profiles for {nb_template_profiles} customers")
        print(f"Batch size: {batch_size}, Chunk size: {chunk_size}, Workers: {max_workers}")
        print(f"Expected throughput: ~{batch_size/seconds_per_batch:.1f} profiles/second")
        
        # Process profiles in batches
        for batch_start in tqdm(range(0, nb_template_profiles, batch_size), desc="Processing batch windows"):
            batch_start_time = time.time()
            
            batch_end = min(batch_start + batch_size, nb_template_profiles)
            batch_profiles = [self.template_profile_dataset[i] for i in range(batch_start, batch_end)]
            
            # Split batch into parallel chunks
            chunks = []
            for chunk_start in range(0, len(batch_profiles), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(batch_profiles))
                chunks.append(batch_profiles[chunk_start:chunk_end])
            
            print(f"Processing {len(batch_profiles)} profiles in {len(chunks)} parallel chunks")
            
            # Process chunks in parallel with ThreadPoolExecutor
            batch_results = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit each chunk as a separate future for true parallelization
                futures = {
                    executor.submit(self._generate_batch_narrative, chunk, debug_prompts): i 
                    for i, chunk in enumerate(chunks)
                }
                
                # Collect results from all parallel chunks
                for future in tqdm(futures, desc=f"Processing chunks {batch_start}-{batch_end}", leave=False):
                    try:
                        chunk_result = future.result()
                        if chunk_result:
                            batch_results.extend(chunk_result)
                    except Exception as e:
                        chunk_idx = futures[future]
                        print(f"Error in chunk {chunk_idx} of batch {batch_start}-{batch_end}: {str(e)}")
            
            # Add all results from this batch window
            narrative_profiles.extend(batch_results)
            
            # Rate limiting between batch windows
            elapsed_time = time.time() - batch_start_time
            if elapsed_time < seconds_per_batch:
                sleep_time = seconds_per_batch - elapsed_time
                print(f"Batch window completed in {elapsed_time:.1f}s, sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
            else:
                print(f"Batch window completed in {elapsed_time:.1f}s (no sleep needed)")
        
        # Optionally append template profiles after batch generation
        if append_template:
            print(f"Appending template profiles to narrative profiles...")
            enhanced_profiles = []
            for profile in narrative_profiles:
                enhanced_profile = {
                    **profile,
                    'narrative_profile': f"{profile['narrative_profile']}\n{profile['template_profile']}"
                }
                enhanced_profiles.append(enhanced_profile)
            self.narrative_profile_dataset = Dataset.from_list(enhanced_profiles)
            print(f"Generated {len(enhanced_profiles)} narrative profiles with templates appended")
        else:
            self.narrative_profile_dataset = Dataset.from_list(narrative_profiles)
            print(f"Generated {len(narrative_profiles)} narrative profiles total")
        
        return self.narrative_profile_dataset
    
    def _generate_batch_narrative(self, batch_profiles: List[Dict[str, Any]], debug_prompts: bool = False) -> List[Dict[str, Any]]:
        """
        Generate narrative profiles for a batch of customers using actual customer IDs.
        
        Args:
            batch_profiles (List[Dict[str, Any]]): Batch of template profiles
            debug_prompts (bool): If True, print the prompt being sent to the LLM
            
        Returns:
            List[Dict[str, Any]]: Batch of profiles with narrative profiles added
        """
        try:
            # Combine template profiles into a single string using actual customer IDs
            batch_template_profiles = []
            customer_ids = []
            
            for i, profile in enumerate(batch_profiles):
                # Get customer ID from profile, fallback to generic ID if not available
                customer_id = profile.get('customer_id', f'customer_{i+1}')
                customer_ids.append(customer_id)
                batch_template_profiles.append(f"Customer {customer_id}: {profile['template_profile']}")
            
            batch_input = "\n\n".join(batch_template_profiles)
            
            # Debug: Print the prompt if requested
            if debug_prompts:
                system_prompt = self.BatchNarrativeProfile.__doc__ or "No system prompt available"
                total_chars = len(system_prompt) + len(self.feature_distribution) + len(batch_input)
                estimated_tokens = total_chars // 4
                
                print(f"\n{'='*80}")
                print(f"PROFILE GENERATION PROMPT DEBUG")
                print(f"{'='*80}")
                print(f"Customer IDs: {', '.join(customer_ids)}")
                print(f"Estimated tokens: ~{estimated_tokens:,}")
                print(f"\nSYSTEM PROMPT:")
                print(system_prompt)
                print(f"\nFEATURE DISTRIBUTION:")
                print(self.feature_distribution)
                print(f"\nCUSTOMER PROFILES BATCH:")
                print(batch_input)
                print(f"{'='*80}\n")
            
            # Generate batch narrative
            result = self.predictor(
                feature_distribution=self.feature_distribution,
                customer_profiles_batch=batch_input
            )
            
            # Parse the batch result as JSON
            raw_response = result.batch_narrative_profiles.strip()
            
            # Try to parse as JSON array
            try:
                import json
                import re
                
                # Try to extract JSON array from response (in case there's extra text)
                json_match = re.search(r'\[.*\]', raw_response, re.DOTALL)
                if json_match:
                    profiles_array = json.loads(json_match.group())
                else:
                    profiles_array = json.loads(raw_response)
                
                # DEBUG: Only print if there's a mismatch
                if len(profiles_array) != len(batch_profiles):
                    print(f"\n{'='*80}")
                    print(f"⚠️  PROFILE MISMATCH: Expected {len(batch_profiles)} profiles, got {len(profiles_array)}")
                    print(f"{'='*80}")
                    print("Raw LLM Response:")
                    print(raw_response)
                    print(f"{'='*80}\n")
                
                # Create a mapping of customer_id to profile for easy lookup
                profile_map = {str(profile.get('customer_id', f'customer_{i+1}')): profile 
                              for i, profile in enumerate(batch_profiles)}
                
                # Match narratives back to profiles by customer_id
                batch_results = []
                matched_ids = set()
                fallback_count = 0
                
                for profile_data in profiles_array:
                    customer_id = str(profile_data.get('customer_id', ''))
                    narrative = profile_data.get('narrative_profile', '')
                    
                    if customer_id in profile_map and narrative:
                        original_profile = profile_map[customer_id]
                        batch_results.append({
                            **original_profile,
                            'narrative_profile': narrative
                        })
                        matched_ids.add(customer_id)
                    else:
                        fallback_count += 1
                
                # Add any missing profiles with template fallback
                for customer_id, profile in profile_map.items():
                    if customer_id not in matched_ids:
                        batch_results.append({
                            **profile,
                            'narrative_profile': profile['template_profile']
                        })
                        fallback_count += 1
                
                # Log fallback summary if any occurred
                if fallback_count > 0:
                    print(f"Warning: {fallback_count}/{len(batch_profiles)} customers used template fallback")
                
                return batch_results
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                # JSON parsing failed - print debug info and use fallback
                print(f"\n{'='*80}")
                print(f"⚠️  JSON PARSING ERROR: {str(e)}")
                print(f"{'='*80}")
                print("Raw LLM Response:")
                print(raw_response)
                print(f"{'='*80}\n")
                print(f"Using template fallback for all {len(batch_profiles)} customers")
                return [{**profile, 'narrative_profile': profile['template_profile']} 
                       for profile in batch_profiles]
            
        except Exception as e:
            # Exception fallback - all customers get template
            print(f"Error in batch processing: {str(e)}")
            print(f"Using template fallback for all {len(batch_profiles)} customers")
            return [{**profile, 'narrative_profile': profile['template_profile']} for profile in batch_profiles]
    
    def save_narrative_profiles(self, output_path: str) -> None:
        """
        Save narrative profiles to file using Polars.
        
        Args:
            output_path (str): Path to save the profiles
        """
        if self.narrative_profile_dataset is None:
            raise ValueError("No narrative profiles generated yet. Call generate_batch_narrative_profiles() first.")
        
        # Convert HuggingFace Dataset to Polars DataFrame
        data_list = [dict(item) for item in self.narrative_profile_dataset]
        df = pl.DataFrame(data_list)
        
        if output_path.endswith('.parquet'):
            df.write_parquet(output_path)
        elif output_path.endswith('.csv'):
            df.write_csv(output_path)
        else:
            raise ValueError("Output path must end with .parquet or .csv")
        
        print(f"Narrative profiles saved to {output_path}")
