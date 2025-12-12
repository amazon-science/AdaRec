# -*- coding: utf-8 -*-
"""
Classification evaluation metrics

This module provides comprehensive evaluation functions for binary classification
models, including metrics calculation and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, auc, brier_score_loss
)

def evaluate_binary_classification(y_true, y_pred_proba, threshold=0.5):
    """
    Comprehensive binary classification evaluation function.
    
    Args:
        y_true: True binary labels (numpy array)
        y_pred_proba: Predicted probabilities for positive class (numpy array)
        threshold: Classification threshold (default 0.5)
    
    Returns:
        Dictionary with all metrics
        
    Examples:
        >>> metrics = evaluate_binary_classification(y_true, y_pred_proba, threshold=0.3)
        >>> print(f"AUC: {metrics['roc_auc']:.3f}, F1: {metrics['f1_score']:.3f}")
    """
    # Convert to numpy arrays if needed
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    
    # Generate binary predictions from probabilities
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'threshold': threshold,
        'classification_report': classification_report(y_true, y_pred),
        'brier_score_loss': brier_score_loss(y_true, y_pred_proba),
    }
    
    # Add precision-recall AUC
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    metrics['pr_auc'] = auc(recall_vals, precision_vals)
    
    return metrics


def plot_confusion_matrix(y_true, y_pred_proba, threshold=0.5, 
                         class_names=['Negative', 'Positive'],
                         figsize=(10, 6), cmap='Blues'):
    """
    Plot a nice confusion matrix with percentages and counts.
    
    Args:
        y_true: True binary labels (numpy array)
        y_pred_proba: Predicted probabilities (numpy array)
        threshold: Classification threshold
        class_names: Names for the classes
        figsize: Figure size tuple
        cmap: Colormap for the heatmap
        
    Examples:
        >>> plot_confusion_matrix(y_true, y_pred_proba, threshold=0.3)
    """
    # Convert to numpy arrays if needed
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    
    # Generate predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(cm, annot=False, fmt='d', cmap=cmap, ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    # Add custom annotations with both count and percentage
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percentage = cm_normalized[i, j] * 100
            ax.text(j + 0.5, i + 0.5, f'{count}\n({percentage:.1f}%)',
                   ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix (Threshold: {threshold})', fontsize=14, fontweight='bold')
    
    # Calculate metrics for display
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    plt.tight_layout()
    plt.show()


def plot_calibration_curve(y_true, y_pred_proba, n_bins=100, strategy='uniform', 
                          figsize=(8, 6), title="Calibration Plot"):
    """
    Plot calibration curve to assess probability calibration.
    
    Args:
        y_true: True binary labels (numpy array)
        y_pred_proba: Predicted probabilities (numpy array)
        n_bins: Number of bins for calibration curve
        strategy: 'quantile' or 'uniform' binning strategy
        figsize: Figure size tuple
        title: Plot title
        
    Examples:
        >>> plot_calibration_curve(y_true, y_pred_proba, n_bins=10)
    """
    from sklearn.calibration import calibration_curve
    
    # Convert to numpy arrays if needed
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    
    # Get calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy=strategy
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot calibration curve
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", 
            label=f"Model ({strategy} bins)", linewidth=2, markersize=8)
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", linewidth=2)
    
    # Customize plot
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add Brier score
    brier_score = brier_score_loss(y_true, y_pred_proba)
    ax.text(0.05, 0.95, f'Brier Score: {brier_score:.4f}', 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return fraction_of_positives, mean_predicted_value
