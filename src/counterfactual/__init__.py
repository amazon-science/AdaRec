"""
Counterfactual generation module for batch processing with DSPy and AWS Bedrock.

This module extends the Potemkin architecture to provide batch counterfactual generation
capabilities for Amazon Basket (BxGy) promotions.
"""

from .batch_profile_generator import BatchProfileGenerator
from .batch_predictor import BatchPredictor
from .base_profile_generator import BaseProfileGenerator
from .util import format_value

__all__ = [
    'BatchProfileGenerator', 
    'BatchPredictor', 
    'BaseProfileGenerator',
    'format_value'
]
