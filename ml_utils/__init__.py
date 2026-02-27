"""
Machine Learning Utilities for ChemometricSolutions
Multi-model Regression + Classification with PyCaret-style comparison
"""

from .ml_analysis import (
    train_multiple_models,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_confusion_matrix,
    plot_feature_importance
)

__all__ = [
    'train_multiple_models',
    'plot_actual_vs_predicted',
    'plot_residuals',
    'plot_confusion_matrix',
    'plot_feature_importance'
]

__version__ = '2.0.0'
__author__ = 'Roquette'