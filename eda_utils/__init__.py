"""
eda_utils — Exploratory Data Analysis for ChemometricSolutions
==============================================================

Minitab-style "Summary Report" for every numeric variable in a dataset.

Each report contains:
  • Histogram with fitted normal curve
  • Boxplot
  • 95 % Confidence Interval plot (Mean & Median)
  • Statistics panel:
      - Anderson-Darling normality test (A², P-Value)
      - Descriptive stats (mean, StDev, variance, skewness, kurtosis, N)
      - 5-number summary (min, Q1, median, Q3, max)
      - 95 % CI for Mean, Median, and StDev

Package Structure
-----------------
eda_calculations  :  Statistical computations (AD test, CIs, descriptives)
eda_plots         :  Plotly figures (plot_summary_report, plot_summary_reports_all)
eda_workspace     :  Streamlit tab renderer, session state, Excel export

Quick Start — inside a Streamlit app
--------------------------------------
>>> from eda_utils import render_eda_tab
>>>
>>> # In your main app, after defining tabs:
>>> with tab_eda:
...     render_eda_tab(st.session_state.uploaded_data)

Quick Start — standalone (no Streamlit)
-----------------------------------------
>>> from eda_utils import plot_summary_report, descriptive_statistics
>>> import pandas as pd
>>>
>>> df = pd.read_csv("my_data.csv")
>>> stats = descriptive_statistics(df["Torque"])
>>> fig   = plot_summary_report(df["Torque"], column_name="Torque")
>>> fig.show()

Integration with main app (home screen tab)
-------------------------------------------
Add to your tab list in the main Streamlit file::

    tab_home, tab_univariate, tab_eda, tab_profiles, ... = st.tabs([
        "🏠 Home", "📈 Univariate", "📊 EDA", "🔀 Row Profiles", ...
    ])

    with tab_eda:
        from eda_utils import render_eda_tab
        render_eda_tab(st.session_state.get("uploaded_data"))
"""

# ── Calculations ──────────────────────────────────────────────
from .eda_calculations import (
    anderson_darling_test,
    ci_for_mean,
    ci_for_median,
    ci_for_stdev,
    descriptive_statistics,
    run_eda_for_all_columns,
)

# ── Plots ─────────────────────────────────────────────────────
from .eda_plots import (
    plot_summary_report,
    plot_summary_reports_all,
)

# ── Workspace / Streamlit ─────────────────────────────────────
from .eda_workspace import (
    render_eda_tab,
    export_eda_results_to_excel,
    save_eda_results_to_session,
    load_eda_results_from_session,
)

# ── Public API ────────────────────────────────────────────────
__all__ = [
    # calculations
    "anderson_darling_test",
    "ci_for_mean",
    "ci_for_median",
    "ci_for_stdev",
    "descriptive_statistics",
    "run_eda_for_all_columns",
    # plots
    "plot_summary_report",
    "plot_summary_reports_all",
    # workspace
    "render_eda_tab",
    "export_eda_results_to_excel",
    "save_eda_results_to_session",
    "load_eda_results_from_session",
]

__version__     = "1.0.0"
__author__      = "ChemometricSolutions"
__description__ = "Minitab-style EDA Summary Reports for chemometric data"
