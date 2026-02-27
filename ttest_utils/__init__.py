"""
Two-Sample t-Test Utilities for Roquette Advanced Analytics
"""

from .ttest import (
    perform_two_sample_ttest,
    plot_minitab_style_comparison,
    calculate_sample_size_tost,
    estimate_pooled_sd,
    perform_capability_analysis,
    plot_capability_histograms,
    plot_hypothesis_test_bar,
    perform_imr_analysis,
    plot_imr_chart,
)

__all__ = [
    'perform_two_sample_ttest',
    'plot_minitab_style_comparison',
    'calculate_sample_size_tost',
    'estimate_pooled_sd',
    'perform_capability_analysis',
    'plot_capability_histograms',
    'plot_hypothesis_test_bar',
    'perform_imr_analysis',
    'plot_imr_chart',
]

__version__ = '4.0.0'
__author__ = 'Roquette Advanced Analytics'
