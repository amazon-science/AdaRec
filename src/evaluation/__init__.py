# -*- coding: utf-8 -*-
"""
Evaluation module for SatoriML.

This module provides functions for evaluating machine learning models,
particularly focused on binary classification tasks common in AB testing.
"""

from .metrics import evaluate_binary_classification, plot_confusion_matrix, plot_calibration_curve

__all__ = ['evaluate_binary_classification', 'plot_confusion_matrix', 'plot_calibration_curve']
