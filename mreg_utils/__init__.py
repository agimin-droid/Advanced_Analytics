"""
Multiple Regression Utilities for Roquette Advanced Analytics
"""

from .mreg import (
    generate_extended_features,
    compute_ols_statistics,
    forward_stepwise,
    fit_model,
    compute_incremental_impact,
    compute_x_regressed_on_others,
    detect_unusual_data,
    check_residual_normality,
    generate_report_card,
    format_equation,
    plot_model_building_sequence,
    plot_incremental_impact,
    plot_x_regressed_on_others,
    plot_pvalue_bar,
    plot_scatter_panels,
    plot_diagnostic_report,
    export_results_to_excel,
)

__all__ = [
    "generate_extended_features",
    "compute_ols_statistics",
    "forward_stepwise",
    "fit_model",
    "compute_incremental_impact",
    "compute_x_regressed_on_others",
    "detect_unusual_data",
    "check_residual_normality",
    "generate_report_card",
    "format_equation",
    "plot_model_building_sequence",
    "plot_incremental_impact",
    "plot_x_regressed_on_others",
    "plot_pvalue_bar",
    "plot_scatter_panels",
    "plot_diagnostic_report",
    "export_results_to_excel",
]

__version__ = "2.0.0"
__author__ = "Roquette Advanced Analytics"
