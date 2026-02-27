"""
EDA Page - Exploratory Data Analysis
=====================================

Streamlit page module for the EDA Summary Report feature.
Follows the same pattern as all other page modules (show() entry point).

Renders Minitab-style Summary Reports for every numeric variable:
  - Histogram with fitted normal curve
  - Boxplot
  - 95% Confidence Interval plot (Mean & Median)
  - Statistics panel (Anderson-Darling test, descriptives, 5-number summary, CIs)
"""

import streamlit as st
import pandas as pd

# Import workspace utilities (same pattern as other pages)
from workspace_utils import display_workspace_dataset_selector

# Import EDA utilities
try:
    from eda_utils import render_eda_tab
    EDA_UTILS_AVAILABLE = True
except ImportError as e:
    EDA_UTILS_AVAILABLE = False
    _EDA_IMPORT_ERROR = str(e)


def show():
    """
    Main entry point called by homepage.py router.
    Matches the pattern of all other page modules.
    """
    st.title("📊 Exploratory Data Analysis")
    st.markdown(
        "Minitab-style **Summary Reports** for every numeric variable — "
        "histogram, boxplot, confidence intervals, and Anderson-Darling normality test."
    )

    # ── Check eda_utils is importable ───────────────────────────
    if not EDA_UTILS_AVAILABLE:
        st.error(
            f"⚠️ `eda_utils` package not found. "
            f"Make sure the `eda_utils/` folder is in the same directory as `homepage.py`.\n\n"
            f"**Error:** `{_EDA_IMPORT_ERROR}`"
        )
        st.info(
            "Expected folder structure:\n"
            "```\n"
            "chemometricsolutions-demo/\n"
            "├── homepage.py\n"
            "├── eda_page.py          ← this file\n"
            "├── eda_utils/           ← required folder\n"
            "│   ├── __init__.py\n"
            "│   ├── eda_calculations.py\n"
            "│   ├── eda_plots.py\n"
            "│   └── eda_workspace.py\n"
            "└── ...\n"
            "```"
        )
        return

    # ── Dataset selector (same widget used by all other pages) ──
    result = display_workspace_dataset_selector(
        label="Select dataset for EDA:",
        key="eda_page_dataset_selector",
        help_text="Choose a dataset from your workspace to analyse",
        show_info=True,
    )

    if result is None:
        # No data loaded — display_workspace_dataset_selector already shows the warning
        return

    dataset_name, dataframe = result

    # ── Filter to numeric columns only & warn if non-numeric exist ──
    numeric_cols = dataframe.select_dtypes(include=["number"]).columns.tolist()
    non_numeric_cols = dataframe.select_dtypes(exclude=["number"]).columns.tolist()

    if not numeric_cols:
        st.error(
            "The selected dataset has **no numeric columns**. "
            "EDA requires at least one numeric variable."
        )
        return

    if non_numeric_cols:
        st.info(
            f"ℹ️ {len(non_numeric_cols)} non-numeric column(s) will be excluded from EDA: "
            f"`{'`, `'.join(non_numeric_cols)}`"
        )

    st.divider()

    # ── Delegate to eda_utils renderer ──────────────────────────
    render_eda_tab(
        dataframe=dataframe,
        key_prefix=f"eda_{dataset_name.replace(' ', '_')}",
    )
