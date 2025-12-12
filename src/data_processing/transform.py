# -*- coding: utf-8 -*-
"""
Data transformation functions

This module contains functions for data preprocessing and feature engineering
using Polars for improved performance and modern syntax.
"""

import logging
import re
from typing import List, Dict, Any, Tuple
import polars as pl
import numpy as np


def convert_to_float(
    df: pl.DataFrame, 
    sample_size: int = 100,
    numeric_threshold: float = 0.7,
    do_not_convert: List = ['customer_id'],
) -> pl.DataFrame:
    """
    Convert all numeric and numeric-string columns to Float64.
    
    This function ensures proper float division in downstream calculations by:
    1. Converting existing numeric types (Int, Decimal, Float) to Float64
    2. Detecting string columns that contain numeric values and converting them
    
    Args:
        df: Input Polars DataFrame
        sample_size: Number of non-null values to sample for string analysis
        numeric_threshold: Fraction of sampled values that must be numeric 
                          to consider a string column as numeric (0.0-1.0)
        do_not_convert: Columns to not convert
        
    Returns:
        DataFrame with numeric columns as Float64
    """
    conversions = []
    
    for col in df.columns:
        #for columns to not convert, just make them all string
        if col in do_not_convert: 
            df = df.with_columns(pl.col(col).cast(pl.Utf8))
            continue

        dtype = df[col].dtype
        if dtype.is_numeric():
            # Already numeric - convert to Float64
            conversions.append(pl.col(col).cast(pl.Float64))
            
        elif dtype == pl.Utf8:  # String column
            # Sample non-null values for analysis
            sample_values = df[col].drop_nulls().head(sample_size).to_list()
            
            if sample_values:
                # Check how many are numeric
                numeric_count = 0
                check_count = min(len(sample_values), sample_size)
                
                for val in sample_values[:check_count]:
                    try:
                        float(val)
                        numeric_count += 1
                    except (ValueError, TypeError):
                        pass
                
                # Convert if above threshold
                if check_count > 0 and numeric_count / check_count >= numeric_threshold:
                    conversions.append(pl.col(col).cast(pl.Float64, strict=False))
                else:
                    conversions.append(pl.col(col))  # Keep as string
            else:
                conversions.append(pl.col(col))  # Keep as string (all nulls)
        else:
            conversions.append(pl.col(col))  # Keep other types as-is
    
    return df.with_columns(conversions)


def impute_cols(df: pl.DataFrame) -> pl.DataFrame:
    """
    Impute missing values in DataFrame columns with intelligent defaults.
    
    Note: This function assumes numeric type conversion has already been done
    via convert_to_float(). It focuses on imputation logic only.
    
    - Recency columns (days_since_last_*): impute with 360 (max recency)
    - Numeric columns: impute with 0
    - String columns: impute with empty string
    
    Args:
        df: Input Polars DataFrame
        
    Returns:
        DataFrame with imputed values
    """
    # Get column names that match recency pattern
    recency_cols = [col for col in df.columns if col.startswith('days_since_last_')]
    
    # Create list of transformations
    transformations = []
    
    # Handle recency columns - impute with 360 if null or > 360
    for col in recency_cols:
        if col in df.columns:
            transformations.append(
                pl.when(pl.col(col).is_null() | (pl.col(col) > 360))
                .then(360.0)
                .otherwise(pl.col(col))
                .alias(col)
            )
    
    # Handle other columns based on their type
    for col in df.columns:
        if col not in recency_cols:  # Skip already handled recency columns
            dtype = df[col].dtype
            if dtype.is_numeric():
                # Numeric column - fill nulls with 0
                transformations.append(pl.col(col).fill_null(0.0))
            elif dtype == pl.Utf8:
                # String column - fill nulls with empty string
                transformations.append(pl.col(col).fill_null(""))
            else:
                # Other types - keep as is
                transformations.append(pl.col(col))
    
    # Apply all transformations
    if transformations:
        df = df.with_columns(transformations)
    
    return df


def winsorize_col(
    df: pl.DataFrame, 
    col_name: str, 
    q: float = 0.99,
    both_sides: bool = False
) -> pl.DataFrame:
    """
    Winsorize (clip) extreme values in a column at specified quantiles.
    
    Args:
        df: Input Polars DataFrame
        col_name: Name of column to winsorize
        q: Quantile threshold (0.99 = 99th percentile)
        both_sides: If True, winsorize both tails. If False, only upper tail.
        
    Returns:
        DataFrame with winsorized column
    """
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in DataFrame")
    
    if both_sides:
        # Two-sided winsorization
        per = (1 - q) / 2
        cutoff_low = df[col_name].quantile(per)
        cutoff_high = df[col_name].quantile(1 - per)
        logging.info(f'Winsorizing {col_name}: cutoffs at {cutoff_low} and {cutoff_high}')
        
        return df.with_columns(
            pl.col(col_name).clip(cutoff_low, cutoff_high)
        )
    else:
        # One-sided winsorization (upper tail only)
        # Only consider positive values for quantile calculation
        cutoff = df.filter(pl.col(col_name) > 0)[col_name].quantile(q)
        logging.info(f'Winsorizing {col_name}: cutoff at {cutoff}')
        
        return df.with_columns(
            pl.when(pl.col(col_name) > cutoff)
            .then(cutoff)
            .otherwise(pl.col(col_name))
            .alias(col_name)
        )


def resample_by_outcome(
    df: pl.DataFrame,
    outcome_col: str,
    negative_per_positive: float = 1.0,
    seed: int = 112,
    stratify_cols: List[str] = None
) -> pl.DataFrame:
    """
    Resample DataFrame to balance binary outcome classes.
    
    Modern Polars implementation of outcome-based resampling for handling
    imbalanced datasets in machine learning pipelines.
    
    Args:
        df: Input Polars DataFrame
        outcome_col: Name of binary outcome column (0/1)
        negative_per_positive: Ratio of negative to positive samples to maintain
        seed: Random seed for reproducible sampling
        stratify_cols: Optional columns to maintain distribution during sampling
        
    Returns:
        Resampled DataFrame with balanced outcome classes
        
    Examples:
        # Basic 1:1 resampling
        balanced_df = resample_by_outcome(df, 'outcome_ops_campaign')
        
        # 2:1 negative to positive ratio
        balanced_df = resample_by_outcome(df, 'outcome_ops_campaign', 
                                        negative_per_positive=2.0)
        
        # Stratified by treatment
        balanced_df = resample_by_outcome(df, 'outcome_ops_campaign',
                                        stratify_cols=['treatment_name'])
    """
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found in DataFrame")
    
    # Separate positive and negative cases
    df_pos = df.filter(pl.col(outcome_col) == 1)
    df_neg = df.filter(pl.col(outcome_col) == 0)
    
    n_pos = df_pos.height
    n_neg_target = int(negative_per_positive * n_pos)
    
    logging.info(f"Resampling {outcome_col}: {n_pos} positive cases, "
                f"sampling {n_neg_target} from {df_neg.height} negative cases")
    
    if n_neg_target > df_neg.height:
        logging.warning(f"Requested {n_neg_target} negative samples but only "
                       f"{df_neg.height} available. Using all negative samples.")
        df_neg_sampled = df_neg
    else:
        if stratify_cols:
            # Stratified sampling to maintain distribution
            df_neg_sampled = _stratified_sample(df_neg, n_neg_target, stratify_cols, seed)
        else:
            # Simple random sampling
            df_neg_sampled = df_neg.sample(n=n_neg_target, seed=seed)
    
    # Combine positive and sampled negative cases
    result_df = pl.concat([df_pos, df_neg_sampled], how="vertical")
    
    # Shuffle the final dataset
    result_df = result_df.sample(fraction=1.0, seed=seed)
    
    logging.info(f"Final dataset: {result_df.height} rows "
                f"({result_df.filter(pl.col(outcome_col) == 1).height} positive, "
                f"{result_df.filter(pl.col(outcome_col) == 0).height} negative)")
    
    return result_df


def _stratified_sample(
    df: pl.DataFrame,
    n_samples: int,
    stratify_cols: List[str],
    seed: int
) -> pl.DataFrame:
    """
    Perform stratified sampling to maintain distribution of stratify_cols.
    
    Args:
        df: DataFrame to sample from
        n_samples: Number of samples to draw
        stratify_cols: Columns to maintain distribution for
        seed: Random seed
        
    Returns:
        Stratified sample DataFrame
    """
    # Calculate proportions for each stratum
    strata_counts = df.group_by(stratify_cols).agg(pl.len().alias('count'))
    total_count = df.height
    
    # Calculate target samples per stratum
    strata_targets = strata_counts.with_columns([
        (pl.col('count') / total_count * n_samples).round().cast(pl.Int32).alias('target_count')
    ])
    
    sampled_dfs = []
    
    for row in strata_targets.iter_rows(named=True):
        # Create filter condition for this stratum
        filter_conditions = [pl.col(col) == row[col] for col in stratify_cols]
        stratum_filter = filter_conditions[0]
        for condition in filter_conditions[1:]:
            stratum_filter = stratum_filter & condition
        
        # Sample from this stratum
        stratum_df = df.filter(stratum_filter)
        target_count = min(row['target_count'], stratum_df.height)
        
        if target_count > 0:
            sampled_stratum = stratum_df.sample(n=target_count, seed=seed)
            sampled_dfs.append(sampled_stratum)
    
    return pl.concat(sampled_dfs, how="vertical") if sampled_dfs else df.head(0)


def train_valid_test_split(
    df: pl.DataFrame,
    train_size: float = 0.6,
    valid_size: float = 0.2,
    test_size: float = 0.2,
    stratify_cols: List[str] = ['treatment_name'],
    seed: int = 112,
    shuffle: bool = True
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split DataFrame into train, validation, and test sets with stratification.
    
    Designed specifically for AB testing scenarios where maintaining treatment
    group balance across splits is crucial for valid model evaluation.
    
    Args:
        df: Input Polars DataFrame
        train_size: Proportion for training set (0.0-1.0)
        valid_size: Proportion for validation set (0.0-1.0)
        test_size: Proportion for test set (0.0-1.0)
        stratify_cols: Columns to stratify by (maintains distribution across splits)
        seed: Random seed for reproducible splits
        shuffle: Whether to shuffle data before splitting
        
    Returns:
        Tuple of (train_df, valid_df, test_df)
        
    Examples:
        # Basic AB testing split maintaining treatment balance
        train, valid, test = train_valid_test_split(df, stratify_cols=['treatment_name'])
        
        # Custom split ratios
        train, valid, test = train_valid_test_split(df, 
                                                  train_size=0.7, 
                                                  valid_size=0.15, 
                                                  test_size=0.15)
        
        # Multiple stratification columns
        train, valid, test = train_valid_test_split(df, 
                                                  stratify_cols=['treatment_name', 'is_current_prime'])
    """
    # Validate split sizes
    if not abs(train_size + valid_size + test_size - 1.0) < 1e-6:
        raise ValueError(f"Split sizes must sum to 1.0, got {train_size + valid_size + test_size}")
    
    if any(size <= 0 for size in [train_size, valid_size, test_size]):
        raise ValueError("All split sizes must be positive")
    
    # Validate stratify columns exist
    missing_cols = [col for col in stratify_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Stratify columns not found in DataFrame: {missing_cols}")
    
    # Shuffle if requested
    if shuffle:
        df = df.sample(fraction=1.0, seed=seed)
    
    # Perform stratified split
    train_df, temp_df = _stratified_split(df, train_size, stratify_cols, seed)
    
    # Calculate remaining split ratio for valid/test
    remaining_size = valid_size + test_size
    valid_ratio = valid_size / remaining_size
    
    valid_df, test_df = _stratified_split(temp_df, valid_ratio, stratify_cols, seed + 1)
    
    # Log split summary
    logging.info(f"Dataset split: Train={train_df.height}, Valid={valid_df.height}, Test={test_df.height}")
    
    # Validate splits maintain stratification
    _validate_split_stratification(df, train_df, valid_df, test_df, stratify_cols)
    
    return train_df, valid_df, test_df


def _stratified_split(
    df: pl.DataFrame,
    first_split_size: float,
    stratify_cols: List[str],
    seed: int
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split DataFrame into two parts while maintaining stratification.
    
    Args:
        df: DataFrame to split
        first_split_size: Proportion for first split (0.0-1.0)
        stratify_cols: Columns to maintain distribution for
        seed: Random seed
        
    Returns:
        Tuple of (first_split_df, second_split_df)
    """
    # Calculate samples needed per stratum
    strata_counts = df.group_by(stratify_cols).agg(pl.len().alias('count'))
    
    strata_targets = strata_counts.with_columns([
        (pl.col('count') * first_split_size).round().cast(pl.Int32).alias('first_split_count')
    ])
    
    first_split_dfs = []
    second_split_dfs = []
    
    for row in strata_targets.iter_rows(named=True):
        # Create filter for this stratum
        filter_conditions = [pl.col(col) == row[col] for col in stratify_cols]
        stratum_filter = filter_conditions[0]
        for condition in filter_conditions[1:]:
            stratum_filter = stratum_filter & condition
        
        # Get stratum data
        stratum_df = df.filter(stratum_filter)
        first_split_count = min(row['first_split_count'], stratum_df.height)
        
        if first_split_count > 0:
            # Sample for first split
            first_split_stratum = stratum_df.sample(n=first_split_count, seed=seed)
            first_split_dfs.append(first_split_stratum)
            
            # Remaining goes to second split
            if stratum_df.height > first_split_count:
                # Get customer IDs from first split to exclude
                if 'customer_id' in stratum_df.columns:
                    first_split_ids = first_split_stratum['customer_id']
                    second_split_stratum = stratum_df.filter(
                        ~pl.col('customer_id').is_in(first_split_ids)
                    )
                else:
                    # Fallback: use row indices (less efficient)
                    second_split_stratum = stratum_df.slice(first_split_count)
                
                second_split_dfs.append(second_split_stratum)
    
    first_split_df = pl.concat(first_split_dfs, how="vertical") if first_split_dfs else df.head(0)
    second_split_df = pl.concat(second_split_dfs, how="vertical") if second_split_dfs else df.head(0)
    
    return first_split_df, second_split_df


def _validate_split_stratification(
    original_df: pl.DataFrame,
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    test_df: pl.DataFrame,
    stratify_cols: List[str],
    tolerance: float = 0.05
) -> None:
    """
    Validate that splits maintain the original stratification distribution.
    
    Args:
        original_df: Original DataFrame
        train_df: Training split
        valid_df: Validation split  
        test_df: Test split
        stratify_cols: Columns that were stratified
        tolerance: Maximum allowed deviation from original proportions
    """
    # Calculate original proportions
    original_props = (original_df.group_by(stratify_cols)
                     .agg(pl.len().alias('count'))
                     .with_columns((pl.col('count') / original_df.height).alias('proportion')))
    
    # Check each split
    for split_name, split_df in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
        if split_df.height == 0:
            continue
            
        split_props = (split_df.group_by(stratify_cols)
                      .agg(pl.len().alias('count'))
                      .with_columns((pl.col('count') / split_df.height).alias('proportion')))
        
        # Join to compare proportions
        comparison = original_props.join(split_props, on=stratify_cols, how='outer', suffix='_split')
        
        # Check for large deviations
        for row in comparison.iter_rows(named=True):
            original_prop = row.get('proportion', 0)
            split_prop = row.get('proportion_split', 0)
            
            if abs(original_prop - split_prop) > tolerance:
                logging.warning(f"Stratification deviation in {split_name} split: "
                               f"{dict((col, row[col]) for col in stratify_cols)} "
                               f"original={original_prop:.3f}, split={split_prop:.3f}")

