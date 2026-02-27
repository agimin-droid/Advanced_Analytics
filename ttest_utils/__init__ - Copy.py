"""
Two-Sample t-Test Utilities for Roquette Advanced Analytics
"""

from .ttest import (
    perform_two_sample_ttest,
    plot_minitab_style_comparison,
    calculate_sample_size_tost,
    estimate_pooled_sd
)

__all__ = [
    'perform_two_sample_ttest',
    'plot_minitab_style_comparison',
    'calculate_sample_size_tost',
    'estimate_pooled_sd'
]

__version__ = '2.0.0'
__author__ = 'Roquette Advanced Analytics'