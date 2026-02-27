"""
Univariate Analysis Utilities for ChemometricSolutions
======================================================

Comprehensive package for univariate statistical analysis including:
- Descriptive statistics (mean, median, geometric mean)
- Dispersion measures (standard deviation, variance, RSD)
- Robust statistics (IQR, MAD, Robust CV)
- Visualizations (histograms, density, boxplots, profiles)
- Row profile analysis for analytical chemometrics
- Two-sample t-test (Minitab-equivalent)

Package Structure
-----------------
univariate_calculations : Descriptive and dispersion statistics
univariate_plots        : Plotly visualization functions
univariate_workspace    : Workspace and dataset management
row_profiles            : Enhanced row profile plotting
ttest_calculations      : Two-sample t-test (Minitab-style)
ttest_plots             : T-test visualizations

Quick Start
-----------
>>> from univariate_utils import (
...     calculate_descriptive_stats,
...     calculate_dispersion_stats,
...     plot_histogram,
...     plot_row_profiles,
...     two_sample_ttest
... )
>>> import pandas as pd
>>> data = pd.DataFrame(...)
>>>
>>> # Calculate statistics for a column
>>> stats = calculate_descriptive_stats(data['Column1'])
>>>
>>> # Plot histogram with custom colors
>>> fig = plot_histogram(data['Column1'], color_palette='viridis')
>>>
>>> # Two-sample t-test
>>> results = two_sample_ttest(data['A'].values, data['B'].values)
"""

# Import calculation functions
from .univariate_calculations import (
    calculate_descriptive_stats,
    calculate_dispersion_stats,
    calculate_robust_stats,
    get_column_statistics_summary,
    get_row_profile_stats
)

# Import plotting functions
from .univariate_plots import (
    plot_histogram,
    plot_density,
    plot_boxplot,
    plot_stripchart,
    plot_eda_plot,
    plot_row_profiles,
    plot_row_profiles_colored
)

# Import enhanced row profiles
from .row_profiles import (
    plot_row_profiles_enhanced
)

# Import workspace functions
from .univariate_workspace import (
    save_univariate_results,
    load_univariate_results,
    export_statistics_to_csv,
    export_statistics_to_excel,
    format_statistics_for_display,
    get_all_saved_results,
    clear_univariate_results
)

# Import two-sample t-test functions
from .ttest_calculations import (
    two_sample_ttest,
    test_equal_variances,
    test_normality,
    ttest_power,
    format_minitab_output,
    pairwise_ttest
)

from .ttest_plots import (
    plot_individual_value,
    plot_comparative_boxplot,
    plot_histogram_overlay,
    plot_qq_by_group,
    plot_interval_plot,
    plot_ttest_fourplot,
    plot_test_report
)


# Define public API
__all__ = [
    # Calculations
    'calculate_descriptive_stats',
    'calculate_dispersion_stats',
    'calculate_robust_stats',
    'get_column_statistics_summary',
    'get_row_profile_stats',

    # Plotting
    'plot_histogram',
    'plot_density',
    'plot_boxplot',
    'plot_stripchart',
    'plot_eda_plot',
    'plot_row_profiles',
    'plot_row_profiles_colored',
    'plot_row_profiles_enhanced',

    # Workspace
    'save_univariate_results',
    'load_univariate_results',
    'export_statistics_to_csv',
    'export_statistics_to_excel',
    'format_statistics_for_display',
    'get_all_saved_results',
    'clear_univariate_results',

    # Two-Sample T-Test
    'two_sample_ttest',
    'test_equal_variances',
    'test_normality',
    'ttest_power',
    'format_minitab_output',
    'pairwise_ttest',
    'plot_individual_value',
    'plot_comparative_boxplot',
    'plot_histogram_overlay',
    'plot_qq_by_group',
    'plot_interval_plot',
    'plot_ttest_fourplot',
    'plot_test_report',
]

__version__ = '1.1.0'
__author__ = 'ChemometricSolutions'
__description__ = 'Univariate analysis utilities for chemometric data'
