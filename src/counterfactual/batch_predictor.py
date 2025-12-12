"""
Batch Predictor for counterfactual analysis.

Takes a HuggingFace dataset and generates predictions by adding new columns
with pred_ prefix based on outcome configuration.
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
import json
import re

from .util import format_value
from .similarity_searcher import SimilaritySearcher
from .batch_profile_generator import BatchProfileGenerator


class BatchPredictor:
    """
    Simplified Batch Predictor that takes a HuggingFace dataset and adds prediction columns.
    
    This class processes customer data in batches to generate counterfactual predictions
    based on customer profiles, treatments, and expected outcomes.
    """
    
    def __init__(
        self,
        generator: BatchProfileGenerator,
        customer_profile_column: str,
        model_config: str,
        outcome_config: str,
        outcome_context: str = None,
        reference_generator: Optional[BatchProfileGenerator] = None,
        similarity_features: Optional[List[str]] = None,
        top_k_examples: int = 5,
        outcome_distribution_exclude_cols: Optional[List[str]] = None
    ):
        """
        Initialize the BatchPredictor.
        
        Args:
            generator (BatchProfileGenerator): BatchProfileGenerator containing both original DataFrame 
                                             and generated profiles dataset for prediction
            customer_profile_column (str): Column name for customer profiles
            model_config (str): Path to model configuration YAML file
            outcome_config (str): Path to outcome configuration YAML file
            outcome_context (str): Contextual description for outcome distribution. 
                                 Set to None to disable outcome distribution generation.
            reference_generator (BatchProfileGenerator, optional): Reference BatchProfileGenerator 
                                                                  for similarity search (can be different from main generator).
            similarity_features (List[str], optional): List of numerical feature names to use
                                                      for similarity search. Required if reference_generator is provided.
            top_k_examples (int): Number of similar examples to retrieve for ICL context.
            outcome_distribution_exclude_cols (List[str], optional): List of outcome column names to exclude 
                                                                   from outcome distribution calculation.
        """
        self.generator = generator
        self.df = generator.df
        self.dataset = generator.narrative_profile_dataset
        self.customer_profile_column = customer_profile_column
        self.outcome_context = outcome_context
        self.top_k_examples = top_k_examples
        
        # Load configurations
        self.outcome_config = self._load_outcome_config(outcome_config)
        self.model_config = self._load_model_config(model_config)
        
        # Initialize similarity searcher if reference generator is provided
        self.similarity_searcher = None
        self.reference_generator = reference_generator
        if reference_generator is not None:
            if similarity_features is None:
                raise ValueError("similarity_features must be provided when reference_generator is specified")
            
            self.similarity_searcher = SimilaritySearcher(
                reference_df=reference_generator.df,
                similarity_features=similarity_features,
                metric='euclidean',
                n_trees=10
            )
        
        # Generate outcome distribution only if outcome_context is provided
        if self.outcome_context is not None:
            self.outcome_distribution = self.describe_outcome_distribution(outcome_distribution_exclude_cols)
        else:
            self.outcome_distribution = None
        
        # Initialize AWS Bedrock client
        try:
            self.bedrock = boto3.client('bedrock-runtime', region_name=self.model_config['bedrock']['region'])
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize AWS Bedrock client. "
                "Ensure AWS credentials are properly configured."
            ) from e
        
        # Configure DSPy with Bedrock
        prediction_model_config = self.model_config['bedrock']['prediction_model']
        dspy.settings.configure(
            lm=dspy.LM(
                model=prediction_model_config['model_id'],
                temperature=prediction_model_config['temperature'],
                max_tokens=prediction_model_config['max_tokens'],
                num_retries=prediction_model_config['num_retries']
            )
        )
        
        # Create dynamic prediction signature based on outcome config
        self.BatchCounterfactualPrediction = self._create_prediction_signature()
        self.predictor = dspy.Predict(self.BatchCounterfactualPrediction)
        
        # Store enhanced dataset
        self.enhanced_dataset = None
    
    
    def _load_outcome_config(self, outcome_config_path: str) -> List[Dict[str, Any]]:
        """Load outcome configuration from YAML file."""
        with open(outcome_config_path, 'r') as f:
            outcome_config = yaml.safe_load(f)
        return outcome_config.get('outcome_features', [])
    
    def _load_model_config(self, model_config_path: str) -> Dict[str, Any]:
        """Load model configuration from YAML file."""
        with open(model_config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def describe_outcome_distribution(self, outcome_distribution_exclude_cols: Optional[List[str]] = None) -> str:
        """
        Generate descriptive statistics for outcome variables specified in outcome_config.
        
        Args:
            outcome_distribution_exclude_cols (List[str], optional): List of outcome column names to exclude 
                                                                   from outcome distribution calculation.
        
        Returns:
            str: Generated paragraphs describing the distribution of each outcome variable.
            - For numeric outcomes: min, max, mean, and all deciles (10th-90th percentiles)
            - For binary outcomes: mean percentage
            - For string outcomes: value counts
        """
        # Default to empty list if None
        outcome_distribution_exclude_cols = outcome_distribution_exclude_cols or []
        
        # Start with the parameterized context
        paragraphs = [self.outcome_context]
        
        # Use the full DataFrame for statistical analysis
        df = self.df
        
        # Generate statistics for each outcome feature
        for outcome_feature in self.outcome_config:
            feature_name = outcome_feature['feature_name']
            feature_description = outcome_feature['feature_description']
            data_type = outcome_feature.get('data_type', 'float')
            
            # Skip if excluded
            if feature_name in outcome_distribution_exclude_cols:
                continue
            
            if feature_name not in df.columns:
                continue
            
            row = df[feature_name]
            
            # Handle string outcomes
            if data_type == 'string':
                value_counts = row.value_counts(sort=True, parallel=True)
                total_count = len(row)
                lines = [f"Distribution of {feature_description}:"]
                lines.append("- Value counts:")
                for item in value_counts.iter_rows():
                    value, count = item
                    percentage = count / total_count
                    lines.append(f"  {value}: {percentage:.2%}")
                feature_text = "\n".join(lines)
                paragraphs.append(feature_text)
            
            # Handle binary outcomes
            elif data_type == 'binary':
                mean = row.mean()
                feature_text = (f"Distribution of {feature_description}:\n"
                            f"- Mean: {mean:.2%}")
                paragraphs.append(feature_text)
            
            # Handle numeric outcomes (int/float)
            else:
                stats = {
                    'min': row.min(),
                    'max': row.max(),
                    'mean': row.mean(),
                }
                
                # Calculate all deciles (10th through 90th percentiles)
                percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                for i, percentile in enumerate(percentiles):
                    p = [10, 20, 30, 40, 50, 60, 70, 80, 90][i]
                    stats[f'p{p}'] = row.quantile(percentile)
                
                # Format all values according to outcome's data type
                formatted_stats = {
                    k: format_value(v, data_type) 
                    for k, v in stats.items()
                }
                
                # Create description paragraph
                lines = [f"Distribution of {feature_description}:"]
                lines.append(f"- Range: {formatted_stats['min']} to {formatted_stats['max']}")
                
                # Calculate non-zero percentage
                non_zero_pct = (row > 0).mean()
                lines.append(f"- Percent of non-zero values: {non_zero_pct:.2%}")
                lines.append(f"- Mean: {formatted_stats['mean']}")
                lines.append("- Percentiles:")
                for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
                    lines.append(f"  {p}th: {formatted_stats[f'p{p}']}")
                
                feature_text = "\n".join(lines)
                paragraphs.append(feature_text)
        
        return "\n\n".join(paragraphs)
    
    def _create_prediction_signature(self):
        """Create dynamic DSPy signature based on outcome configuration."""
        
        # Create output field descriptions based on outcome config
        output_descriptions = []
        for outcome_feature in self.outcome_config:
            feature_name = outcome_feature['feature_name']
            description = outcome_feature['feature_description']
            data_type = outcome_feature.get('data_type', 'float')
            output_descriptions.append(f'pred_{feature_name}: Predicted {description.lower()} ({data_type})')
        
        # Add standard prediction fields
        output_descriptions.extend([
            'pred_confidence: Confidence level in the prediction (float, 0-1)',
            'pred_reasoning: Detailed explanation for the prediction (string)'
        ])
        
        # Format for the LLM
        output_format = 'JSON format with fields: {' + ', '.join([f'"{desc.split(":")[0]}": {desc.split(":")[1].strip()}' for desc in output_descriptions]) + '}'
        
        # Create different signatures based on available context
        has_similarity = self.similarity_searcher is not None
        has_distribution = self.outcome_distribution is not None
        
        # Create JSON array output description
        json_output_desc = f'CRITICAL: Output a valid JSON array of predictions. Each object must include "customer_id" (string) and all prediction fields: {output_format}. Example: [{{"customer_id": "123", "pred_outcome_upp_campaign": 2.5, ...}}, {{"customer_id": "456", ...}}]. You MUST include ALL customers from the input. The output must be valid JSON that can be parsed with json.loads().'
        
        if has_distribution and has_similarity:
            class DynamicCounterfactualPrediction(dspy.Signature):
                """Generate counterfactual predictions for a batch of customers."""
                outcome_distribution: str = dspy.InputField(desc='Statistical distribution of outcome variables in the dataset')
                similar_examples: str = dspy.InputField(desc='Similar customer examples with actual outcomes for reference. Use these to ground your predictions.')
                customer_profiles_batch: str = dspy.InputField(desc='Batch of customer profiles with treatments, one per line with format "Customer [ID]: Profile: ... Treatment: ..."')
                batch_predictions: str = dspy.OutputField(desc=json_output_desc)
        elif has_distribution:
            class DynamicCounterfactualPrediction(dspy.Signature):
                """Generate counterfactual predictions for a batch of customers."""
                outcome_distribution: str = dspy.InputField(desc='Statistical distribution of outcome variables in the dataset')
                customer_profiles_batch: str = dspy.InputField(desc='Batch of customer profiles with treatments, one per line with format "Customer [ID]: Profile: ... Treatment: ..."')
                batch_predictions: str = dspy.OutputField(desc=json_output_desc)
        elif has_similarity:
            class DynamicCounterfactualPrediction(dspy.Signature):
                """Generate counterfactual predictions for a batch of customers."""
                similar_examples: str = dspy.InputField(desc='Similar customer examples with actual outcomes for reference. Use these to ground your predictions.')
                customer_profiles_batch: str = dspy.InputField(desc='Batch of customer profiles with treatments, one per line with format "Customer [ID]: Profile: ... Treatment: ..."')
                batch_predictions: str = dspy.OutputField(desc=json_output_desc)
        else:
            class DynamicCounterfactualPrediction(dspy.Signature):
                """Generate counterfactual predictions for a batch of customers."""
                customer_profiles_batch: str = dspy.InputField(desc='Batch of customer profiles with treatments, one per line with format "Customer [ID]: Profile: ... Treatment: ..."')
                batch_predictions: str = dspy.OutputField(desc=json_output_desc)
        
        # Set system prompt from config
        system_prompt = self.model_config['prompts']['counterfactual_prediction']['system_prompt']
        DynamicCounterfactualPrediction.__doc__ = system_prompt
        
        return DynamicCounterfactualPrediction
    
    def generate_predictions(
        self,
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None,
        seconds_per_batch: Optional[int] = None,
        chunk_size: Optional[int] = None,
        debug_prompts: bool = False
    ) -> Dataset:
        """
        Generate counterfactual predictions for all customers in the dataset.
        
        Args:
            batch_size (int, optional): Number of customers per batch window
            max_workers (int, optional): Number of concurrent workers
            seconds_per_batch (int, optional): Minimum seconds between batch windows
            chunk_size (int, optional): Size of parallel chunks within each batch
            debug_prompts (bool): If True, print the actual prompts being sent to the LLM for debugging
        
        Returns:
            Dataset: Enhanced dataset with prediction columns added
        """
        # Use config defaults if not specified
        batch_size = batch_size or self.model_config['batch_processing']['prediction_batch_size']
        max_workers = max_workers or self.model_config['batch_processing']['prediction_max_workers']
        seconds_per_batch = seconds_per_batch or self.model_config['batch_processing']['prediction_seconds_per_batch']
        
        # Calculate optimal chunk size for parallel processing
        if chunk_size is None:
            chunk_size = max(1, batch_size // max_workers)
        
        all_predictions = []
        nb_customers = len(self.dataset)
        
        print(f"Generating counterfactual predictions for {nb_customers} customers")
        print(f"Batch size: {batch_size}, Chunk size: {chunk_size}, Workers: {max_workers}")
        print(f"Expected throughput: ~{batch_size/seconds_per_batch:.1f} predictions/second")
        
        # Process customers in batches
        for batch_start in tqdm(range(0, nb_customers, batch_size), desc="Processing batch windows"):
            batch_start_time = time.time()
            
            batch_end = min(batch_start + batch_size, nb_customers)
            batch_customers = [self.dataset[i] for i in range(batch_start, batch_end)]
            
            # Split batch into parallel chunks
            chunks = []
            for chunk_start in range(0, len(batch_customers), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(batch_customers))
                chunks.append(batch_customers[chunk_start:chunk_end])
            
            print(f"Processing {len(batch_customers)} customers in {len(chunks)} parallel chunks")
            
            # Process chunks in parallel with ThreadPoolExecutor
            batch_results = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit each chunk as a separate future for true parallelization
                futures = {
                    executor.submit(self._generate_batch_predictions, chunk, debug_prompts): i 
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
            all_predictions.extend(batch_results)
            
            # Rate limiting between batch windows
            elapsed_time = time.time() - batch_start_time
            if elapsed_time < seconds_per_batch:
                sleep_time = seconds_per_batch - elapsed_time
                print(f"Batch window completed in {elapsed_time:.1f}s, sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
            else:
                print(f"Batch window completed in {elapsed_time:.1f}s (no sleep needed)")
        
        # Enhance original dataset with predictions
        self.enhanced_dataset = self._add_predictions_to_dataset(all_predictions)
        print(f"✅ Generated predictions for {len(all_predictions)} customers")
        return self.enhanced_dataset
    
    def _generate_batch_predictions(self, batch_customers: List[Dict[str, Any]], debug_prompts: bool = False) -> List[Dict[str, Any]]:
        """
        Generate predictions for a batch of customers.
        
        Args:
            batch_customers (List[Dict[str, Any]]): Batch of customer data
            debug_prompts (bool): If True, print the prompt being sent to the LLM
            
        Returns:
            List[Dict[str, Any]]: Batch of customers with predictions added
        """
        try:
            # Combine customer profiles and treatments into a single string
            batch_profiles = []
            customer_ids = []
            
            for i, customer in enumerate(batch_customers):
                customer_id = customer.get('customer_id', f'customer_{i+1}')
                customer_ids.append(customer_id)
                
                profile = customer.get(self.customer_profile_column, 'N/A')
                treatment = customer.get('template_treatment', 'N/A')
                
                combined_profile = f"Customer {customer_id}:\nProfile: {profile}\nTreatment: {treatment}"
                batch_profiles.append(combined_profile)
            
            batch_input = "\n\n".join(batch_profiles)
            
            # Retrieve similar examples if similarity searcher is available
            similar_examples_text = None
            if self.similarity_searcher is not None:
                similar_examples_text = self._format_similar_examples(batch_customers)
            
            # Debug: Print the prompt if requested
            if debug_prompts:
                system_prompt = self.BatchCounterfactualPrediction.__doc__ or "No system prompt available"
                
                # Calculate total prompt size
                prompt_parts = [system_prompt, batch_input]
                if self.outcome_distribution is not None:
                    prompt_parts.append(self.outcome_distribution)
                if similar_examples_text is not None:
                    prompt_parts.append(similar_examples_text)
                
                total_chars = sum(len(part) for part in prompt_parts)
                estimated_tokens = total_chars // 4
                
                print(f"\n{'='*80}")
                print(f"COUNTERFACTUAL PREDICTION PROMPT DEBUG")
                print(f"{'='*80}")
                print(f"Customer IDs: {', '.join(customer_ids)}")
                print(f"Estimated tokens: ~{estimated_tokens:,}")
                print(f"Has outcome distribution: {self.outcome_distribution is not None}")
                print(f"Has similar examples: {similar_examples_text is not None}")
                print(f"\nSYSTEM PROMPT:")
                print(system_prompt)
                
                if self.outcome_distribution is not None:
                    print(f"\nOUTCOME DISTRIBUTION:")
                    print(self.outcome_distribution)
                
                if similar_examples_text is not None:
                    print(f"\nSIMILAR EXAMPLES:")
                    print(similar_examples_text)
                
                print(f"\nCUSTOMER PROFILES BATCH:")
                print(batch_input)
                print(f"{'='*80}\n")
            
            # Generate batch predictions - handle different combinations of context
            has_distribution = self.outcome_distribution is not None
            has_similarity = similar_examples_text is not None
            
            if has_distribution and has_similarity:
                result = self.predictor(
                    outcome_distribution=self.outcome_distribution,
                    similar_examples=similar_examples_text,
                    customer_profiles_batch=batch_input
                )
            elif has_distribution:
                result = self.predictor(
                    outcome_distribution=self.outcome_distribution,
                    customer_profiles_batch=batch_input
                )
            elif has_similarity:
                result = self.predictor(
                    similar_examples=similar_examples_text,
                    customer_profiles_batch=batch_input
                )
            else:
                result = self.predictor(
                    customer_profiles_batch=batch_input
                )
            
            # Parse the batch result as JSON array
            raw_response = result.batch_predictions.strip()
            
            # Try to parse as JSON array
            try:
                # Try to extract JSON array from response (in case there's extra text)
                json_match = re.search(r'\[.*\]', raw_response, re.DOTALL)
                if json_match:
                    predictions_array = json.loads(json_match.group())
                else:
                    predictions_array = json.loads(raw_response)
                
                # DEBUG: Only print if there's a mismatch
                if len(predictions_array) != len(batch_customers):
                    print(f"\n{'='*80}")
                    print(f"⚠️  PREDICTION MISMATCH: Expected {len(batch_customers)} predictions, got {len(predictions_array)}")
                    print(f"{'='*80}")
                    print("Raw LLM Response:")
                    print(raw_response)
                    print(f"{'='*80}\n")
                
                # Create a mapping of customer_id to customer for easy lookup
                customer_map = {str(customer.get('customer_id', f'customer_{i+1}')): customer 
                               for i, customer in enumerate(batch_customers)}
                
                # Match predictions back to customers by customer_id
                batch_results = []
                matched_ids = set()
                fallback_count = 0
                
                for pred_data in predictions_array:
                    customer_id = str(pred_data.get('customer_id', ''))
                    
                    if customer_id in customer_map:
                        original_customer = customer_map[customer_id]
                        
                        # Validate and extract predictions
                        try:
                            predictions = self._validate_predictions(pred_data)
                            batch_results.append({**original_customer, **predictions})
                            matched_ids.add(customer_id)
                        except Exception as e:
                            print(f"Error validating prediction for customer {customer_id}: {str(e)}")
                            predictions = self._generate_fallback_predictions()
                            batch_results.append({**original_customer, **predictions})
                            matched_ids.add(customer_id)
                            fallback_count += 1
                    else:
                        fallback_count += 1
                
                # Add any missing customers with fallback predictions
                for customer_id, customer in customer_map.items():
                    if customer_id not in matched_ids:
                        print(f"Missing prediction for customer {customer_id}")
                        predictions = self._generate_fallback_predictions()
                        batch_results.append({**customer, **predictions})
                        fallback_count += 1
                
                # Log fallback summary if any occurred
                if fallback_count > 0:
                    print(f"Warning: {fallback_count}/{len(batch_customers)} customers used fallback predictions")
                
                return batch_results
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                # JSON parsing failed - print debug info and use fallback
                print(f"\n{'='*80}")
                print(f"⚠️  JSON PARSING ERROR: {str(e)}")
                print(f"{'='*80}")
                print("Raw LLM Response:")
                print(raw_response)
                print(f"{'='*80}\n")
                print(f"Using fallback predictions for all {len(batch_customers)} customers")
                fallback_predictions = self._generate_fallback_predictions()
                return [{**customer, **fallback_predictions} for customer in batch_customers]
            
        except Exception as e:
            # Exception fallback - all customers get fallback predictions
            print(f"Error in batch prediction: {str(e)}")
            print(f"Using fallback predictions for all {len(batch_customers)} customers")
            fallback_predictions = self._generate_fallback_predictions()
            return [{**customer, **fallback_predictions} for customer in batch_customers]
    
    def _format_similar_examples(self, batch_customers: List[Dict[str, Any]],
                                 max_example_profile_length: int = 200 ) -> str:
        """
        Retrieve and format similar examples for a batch of customers.
        
        Args:
            batch_customers (List[Dict[str, Any]]): Batch of customer data
            
        Returns:
            str: Formatted similar examples text for LLM context
        """
        # Convert batch_customers to the format expected by similarity searcher
        query_items = []
        
        for customer in batch_customers:
            customer_id = customer.get('customer_id')
            matching_rows = self.df.filter(pl.col('customer_id') == customer_id)
            
            if len(matching_rows) > 0:
                row_dict = matching_rows.to_dicts()[0]
                
                missing_features = []
                for feature in self.similarity_searcher.similarity_features:
                    if feature not in row_dict:
                        missing_features.append(feature)
                
                if missing_features:
                    raise ValueError(
                        f"Missing similarity features in DataFrame for customer {customer_id}: {missing_features}. "
                        f"Available columns: {list(row_dict.keys())}"
                    )
                
                query_items.append(row_dict)
            else:
                raise ValueError(
                    f"Customer {customer_id} not found in original DataFrame (self.df). "
                    f"Cannot perform similarity search without corresponding row data."
                )
        
        # Retrieve similar examples for each customer in the batch
        similar_examples_list = self.similarity_searcher.batch_search(
            query_items=query_items,
            top_k=self.top_k_examples,
            include_distances=True
        )
        
        # Create lookup dictionary for template data from reference generator
        template_data_lookup = {}
        if self.reference_generator is not None:
            for item in self.reference_generator.narrative_profile_dataset:
                customer_id = item['customer_id']
                template_data_lookup[customer_id] = item
        
        # Format examples for each customer
        formatted_sections = []
        
        for i, (customer, similar_examples) in enumerate(zip(batch_customers, similar_examples_list)):
            customer_id = customer.get('customer_id', f'customer_{i+1}')
            section_lines = [f"Similar Examples for Customer {customer_id}:"]
            
            for j, example in enumerate(similar_examples, 1):
                similarity_score = example.get('_similarity_score', 0.0)
                example_customer_id = example.get('customer_id')
                
                # Look up template data from reference generator
                template_data = template_data_lookup.get(example_customer_id, {})
                
                # Get template fields from the lookup
                example_profile = template_data.get(self.customer_profile_column, 
                                                  template_data.get('template_profile', 'N/A'))
                example_treatment = template_data.get('template_treatment', 'N/A')
                example_outcome = template_data.get('template_outcome', 'N/A')
                
                # Build example text
                example_text = [
                    f"\nExample {j} (similarity: {similarity_score:.3f}):",
                    f"Profile: {example_profile}",
                    f"Treatment: {example_treatment}",
                    f"Actual Outcomes: {example_outcome}",
                ]
                section_lines.extend(example_text)

            formatted_sections.append("\n".join(section_lines))
        
        # Join all customer sections with natural separation
        return "\n\n".join(formatted_sections)
    
    def _validate_predictions(self, prediction_json: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean prediction values."""
        validated = {}
        
        # Validate outcome predictions
        for outcome_feature in self.outcome_config:
            feature_name = outcome_feature['feature_name']
            pred_field = f'pred_{feature_name}'
            data_type = outcome_feature.get('data_type', 'float')
            
            raw_value = prediction_json.get(pred_field)
            
            if data_type == 'float':
                try:
                    validated[pred_field] = float(raw_value) if raw_value is not None else 0.0
                except (ValueError, TypeError):
                    validated[pred_field] = 0.0
            elif data_type == 'int':
                try:
                    validated[pred_field] = int(raw_value) if raw_value is not None else 0
                except (ValueError, TypeError):
                    validated[pred_field] = 0
            else:
                validated[pred_field] = str(raw_value) if raw_value is not None else ''
        
        # Validate standard fields
        try:
            confidence = float(prediction_json.get('pred_confidence', 0.5))
            validated['pred_confidence'] = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            validated['pred_confidence'] = 0.5
        
        validated['pred_reasoning'] = str(prediction_json.get('pred_reasoning', 'No reasoning provided'))
        
        return validated
    
    def _generate_fallback_predictions(self) -> Dict[str, Any]:
        """Generate fallback predictions when parsing fails."""
        fallback = {}
        
        # Generate fallback for outcome predictions
        for outcome_feature in self.outcome_config:
            feature_name = outcome_feature['feature_name']
            pred_field = f'pred_{feature_name}'
            data_type = outcome_feature.get('data_type', 'float')
            
            if data_type == 'float':
                fallback[pred_field] = 0.0
            elif data_type == 'int':
                fallback[pred_field] = 0
            else:
                fallback[pred_field] = ''
        
        # Standard fallback fields
        fallback['pred_confidence'] = 0.0
        fallback['pred_reasoning'] = 'Fallback prediction due to processing error'
        
        return fallback
    
    def _add_predictions_to_dataset(self, predictions: List[Dict[str, Any]]) -> Dataset:
        """Add prediction columns to the original dataset."""
        # Convert to list of dictionaries for easier manipulation
        enhanced_data = []
        
        for i, prediction in enumerate(predictions):
            if i < len(self.dataset):
                # Combine original data with predictions
                original_data = dict(self.dataset[i])
                enhanced_data.append({**original_data, **prediction})
            else:
                enhanced_data.append(prediction)
        
        return Dataset.from_list(enhanced_data)
    
    def get_predictions(self) -> Dataset:
        """Get the enhanced dataset with predictions."""
        return self.enhanced_dataset
    
    def save_predictions(self, output_path: str) -> None:
        """Save enhanced dataset to file using Polars."""
        if self.enhanced_dataset is None:
            raise ValueError("No predictions generated yet. Call generate_predictions() first.")
        
        # Convert HuggingFace Dataset to Polars DataFrame
        data_list = [dict(item) for item in self.enhanced_dataset]
        df = pl.DataFrame(data_list)
        
        if output_path.endswith('.parquet'):
            df.write_parquet(output_path)
        elif output_path.endswith('.csv'):
            df.write_csv(output_path)
        else:
            raise ValueError("Output path must end with .parquet or .csv")
        
        print(f"Enhanced dataset with predictions saved to {output_path}")
