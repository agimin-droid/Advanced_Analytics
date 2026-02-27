"""
EDA Workspace Module
====================

Session-state management, export helpers, and the Streamlit tab renderer
for the EDA Summary Report feature.

Key public functions
--------------------
render_eda_tab(dataframe, key_prefix)   → renders the full Streamlit EDA tab
export_eda_results_to_excel(results)    → BytesIO Excel workbook
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional

from .eda_calculations import run_eda_for_all_columns
from .eda_plots import plot_summary_report, plot_summary_reports_all


# ─────────────────────────────────────────────────────────────
#  STREAMLIT TAB RENDERER
# ─────────────────────────────────────────────────────────────

def render_eda_tab(
    dataframe: pd.DataFrame,
    key_prefix: str = "eda",
    metadata_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Render the complete EDA Summary Report tab inside a Streamlit app.

    Intended to be called from the main app like::

        with tab_eda:
            render_eda_tab(st.session_state.uploaded_data)

    Layout
    ------
    - Sidebar / top controls: variable selector, bin count
    - Main area: Minitab-style Summary Report (histogram, boxplot, CI, stats)
    - Expandable: statistics table
    - Download button for Excel export of all variables

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataset to analyse (rows = samples, columns = variables).
    key_prefix : str
        Prefix for Streamlit widget keys (avoids conflicts if multiple tabs).
    metadata_df : pd.DataFrame, optional
        Optional metadata shown in a collapsible section.
    """
    st.subheader("📊 Exploratory Data Analysis — Summary Report")
    st.caption(
        "Minitab-style summary reports for every numeric variable. "
        "Includes Anderson-Darling normality test, descriptive statistics, "
        "and 95 % confidence intervals."
    )

    # ── Identify numeric columns ────────────────────────────────
    numeric_cols = list(dataframe.select_dtypes(include=[np.number]).columns)
    if not numeric_cols:
        st.error("No numeric columns found in the dataset.")
        return

    # ── Controls ────────────────────────────────────────────────
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])

    with col_ctrl1:
        view_mode = st.radio(
            "View mode",
            options=["Single variable", "All variables (scroll)"],
            horizontal=True,
            key=f"{key_prefix}_view_mode",
        )

    with col_ctrl2:
        n_bins = st.slider(
            "Histogram bins",
            min_value=5, max_value=100, value=20, step=5,
            key=f"{key_prefix}_bins",
        )

    with col_ctrl3:
        confidence = st.selectbox(
            "Confidence level",
            options=[0.90, 0.95, 0.99],
            index=1,
            format_func=lambda x: f"{int(x*100)} %",
            key=f"{key_prefix}_conf",
        )

    st.divider()

    # ── SINGLE VARIABLE mode ────────────────────────────────────
    if view_mode == "Single variable":
        selected_col = st.selectbox(
            "Select variable",
            options=numeric_cols,
            key=f"{key_prefix}_col_select",
        )

        # Compute stats (cached in session state to avoid recomputing)
        cache_key = f"{key_prefix}_stats_{selected_col}"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = None  # will be computed below

        with st.spinner(f"Computing statistics for **{selected_col}**…"):
            stats_dict = _get_or_compute_stats(dataframe, selected_col, cache_key)

        # Plot
        fig = plot_summary_report(
            dataframe[selected_col],
            column_name=selected_col,
            stats_dict=stats_dict,
            confidence=confidence,
            n_bins=n_bins,
            height=640,
            width=None,   # responsive
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Expandable stats table ───────────────────────────────
        with st.expander("📋 Full statistics table", expanded=False):
            st.dataframe(
                _stats_to_dataframe(stats_dict, selected_col, confidence),
                use_container_width=True,
                hide_index=True,
            )

    # ── ALL VARIABLES mode ──────────────────────────────────────
    else:
        st.info(
            f"Generating summary reports for **{len(numeric_cols)}** variables. "
            "Scroll down to browse all."
        )

        # Optional column filter
        cols_to_show = st.multiselect(
            "Filter variables (leave empty = show all)",
            options=numeric_cols,
            default=[],
            key=f"{key_prefix}_multi_select",
        )
        if not cols_to_show:
            cols_to_show = numeric_cols

        for col in cols_to_show:
            cache_key = f"{key_prefix}_stats_{col}"
            stats_dict = _get_or_compute_stats(dataframe, col, cache_key)

            fig = plot_summary_report(
                dataframe[col],
                column_name=col,
                stats_dict=stats_dict,
                confidence=confidence,
                n_bins=n_bins,
                height=620,
                width=None,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.divider()

    # ── Download all stats as Excel ─────────────────────────────
    st.markdown("---")
    st.markdown("#### 💾 Export")
    col_dl1, col_dl2 = st.columns([1, 3])
    with col_dl1:
        if st.button("Generate Excel report", key=f"{key_prefix}_gen_excel"):
            with st.spinner("Computing statistics for all variables…"):
                all_stats = run_eda_for_all_columns(dataframe)
            excel_buf = export_eda_results_to_excel(all_stats, dataframe)
            st.session_state[f"{key_prefix}_excel_buf"] = excel_buf

    if f"{key_prefix}_excel_buf" in st.session_state:
        st.download_button(
            label="⬇️  Download Excel report",
            data=st.session_state[f"{key_prefix}_excel_buf"],
            file_name=f"EDA_Summary_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{key_prefix}_dl_excel",
        )


# ─────────────────────────────────────────────────────────────
#  EXPORT HELPERS
# ─────────────────────────────────────────────────────────────

def export_eda_results_to_excel(
    all_stats: Dict[str, Dict[str, Any]],
    dataframe: Optional[pd.DataFrame] = None,
) -> io.BytesIO:
    """
    Export EDA statistics for all columns to a multi-sheet Excel workbook.

    Sheets
    ------
    - ``Summary``     : one row per variable, all statistics as columns
    - ``Raw Data``    : original dataframe (if provided)
    - ``Metadata``    : generation timestamp, software info

    Parameters
    ----------
    all_stats : dict
        Output of ``run_eda_for_all_columns``
    dataframe : pd.DataFrame, optional
        Original data to include as raw data sheet

    Returns
    -------
    BytesIO
        In-memory Excel file ready for ``st.download_button``
    """
    summary_rows = []
    for col, s in all_stats.items():
        if 'error' in s:
            summary_rows.append({'Variable': col, 'Error': s['error']})
            continue
        summary_rows.append({
            'Variable':       col,
            'N':              s['n'],
            'Missing':        s['n_missing'],
            'Mean':           s['mean'],
            'StDev':          s['stdev'],
            'Variance':       s['variance'],
            'Skewness':       s['skewness'],
            'Kurtosis':       s['kurtosis'],
            'Minimum':        s['minimum'],
            'Q1':             s['q1'],
            'Median':         s['median'],
            'Q3':             s['q3'],
            'Maximum':        s['maximum'],
            'CI_Mean_Lower':  s['ci_mean'][0],
            'CI_Mean_Upper':  s['ci_mean'][1],
            'CI_Median_Lower':s['ci_median'][0],
            'CI_Median_Upper':s['ci_median'][1],
            'CI_StDev_Lower': s['ci_stdev'][0],
            'CI_StDev_Upper': s['ci_stdev'][1],
            'AD_Statistic':   s['ad_statistic'],
            'AD_P':           s['ad_p_label'],
            'AD_Reject_H0':   s['ad_reject'],
        })

    summary_df = pd.DataFrame(summary_rows)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        if dataframe is not None:
            # Limit raw data to first 10 000 rows to keep file size reasonable
            dataframe.iloc[:10_000].to_excel(
                writer, sheet_name='Raw Data', index=True
            )

        meta_df = pd.DataFrame({
            'Property': [
                'Report Type',
                'Generated',
                'Variables Analysed',
                'Software',
            ],
            'Value': [
                'EDA Summary Report',
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                len(all_stats),
                'Roquette Analytics — eda_utils',
            ],
        })
        meta_df.to_excel(writer, sheet_name='Metadata', index=False)

    buf.seek(0)
    return buf


def save_eda_results_to_session(
    results: Dict[str, Dict[str, Any]],
    session_key: str = 'eda_results',
) -> None:
    """Store EDA results dict in Streamlit session state."""
    st.session_state[session_key] = {
        'data': results,
        'timestamp': datetime.now().isoformat(),
    }


def load_eda_results_from_session(
    session_key: str = 'eda_results',
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Retrieve EDA results from Streamlit session state."""
    entry = st.session_state.get(session_key)
    return entry['data'] if entry else None


# ─────────────────────────────────────────────────────────────
#  INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────

def _get_or_compute_stats(
    dataframe: pd.DataFrame,
    col: str,
    cache_key: str,
) -> Dict[str, Any]:
    """Return cached stats or compute and cache them."""
    from .eda_calculations import descriptive_statistics

    if st.session_state.get(cache_key) is None:
        st.session_state[cache_key] = descriptive_statistics(dataframe[col])
    return st.session_state[cache_key]


def _stats_to_dataframe(
    s: Dict[str, Any],
    col_name: str,
    confidence: float,
) -> pd.DataFrame:
    """Convert a statistics dict to a tidy display DataFrame."""
    ci_pct = int(confidence * 100)
    rows = [
        ('Normality',  'Anderson-Darling A²',        s['ad_statistic']),
        ('Normality',  'Anderson-Darling P-Value',   s['ad_p_label']),
        ('Normality',  'Normal distribution?',       'No' if s['ad_reject'] else 'Yes'),
        ('Descriptive','N',                           s['n']),
        ('Descriptive','Missing',                     s['n_missing']),
        ('Descriptive','Mean',                        s['mean']),
        ('Descriptive','StDev',                       s['stdev']),
        ('Descriptive','Variance',                    s['variance']),
        ('Descriptive','Skewness',                    s['skewness']),
        ('Descriptive','Kurtosis',                    s['kurtosis']),
        ('5-Number',   'Minimum',                     s['minimum']),
        ('5-Number',   '1st Quartile (Q1)',           s['q1']),
        ('5-Number',   'Median',                      s['median']),
        ('5-Number',   '3rd Quartile (Q3)',           s['q3']),
        ('5-Number',   'Maximum',                     s['maximum']),
        (f'{ci_pct}% CI', 'Mean Lower',              s['ci_mean'][0]),
        (f'{ci_pct}% CI', 'Mean Upper',              s['ci_mean'][1]),
        (f'{ci_pct}% CI', 'Median Lower',            s['ci_median'][0]),
        (f'{ci_pct}% CI', 'Median Upper',            s['ci_median'][1]),
        (f'{ci_pct}% CI', 'StDev Lower',             s['ci_stdev'][0]),
        (f'{ci_pct}% CI', 'StDev Upper',             s['ci_stdev'][1]),
    ]
    return pd.DataFrame(rows, columns=['Category', 'Statistic', 'Value'])
