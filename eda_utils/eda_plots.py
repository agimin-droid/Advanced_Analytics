"""
EDA Plots Module
================

Minitab-style "Summary Report" figures for every numeric variable.

Main function
-------------
plot_summary_report(data, column_name, stats_dict=None) -> go.Figure

Layout (matches the reference image)
--------------------------------------
  ┌──────────────────┬────────────────────┐
  │  Histogram +     │  Statistics panel  │
  │  Normal curve    │  (AD test, means,  │
  ├──────────────────┤   quartiles, CIs)  │
  │  Boxplot         │                    │
  ├──────────────────┤                    │
  │  95% CI plot     │                    │
  │  (Mean & Median) │                    │
  └──────────────────┴────────────────────┘
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sp_stats
from typing import Optional, Dict, Any, Union, List

from .eda_calculations import descriptive_statistics, _clean


# ──────────────────────────────────────────────
#  COLOUR PALETTE  (matches univariate_utils style)
# ──────────────────────────────────────────────
_HIST_BAR   = 'rgba(100, 149, 237, 0.70)'   # cornflower blue
_HIST_LINE  = 'rgba(100, 149, 237, 1.00)'
_NORM_CURVE = 'rgba(200, 50, 50, 0.90)'     # red (matches Minitab)
_BOX_COLOR  = 'rgba(100, 149, 237, 0.60)'
_CI_DOT     = '#1f4e79'
_CI_LINE    = '#2e75b6'
_PANEL_BG   = '#f8f9fa'


# ──────────────────────────────────────────────
#  PUBLIC API
# ──────────────────────────────────────────────

def plot_summary_report(
    data: Union[np.ndarray, pd.Series],
    column_name: str = "Variable",
    stats_dict: Optional[Dict[str, Any]] = None,
    confidence: float = 0.95,
    n_bins: int = 20,
    height: int = 620,
    width: int = 900,
) -> go.Figure:
    """
    Minitab-style Summary Report for a single variable.

    Panels (left column, top → bottom):
        1. Histogram with fitted normal curve
        2. Boxplot
        3. 95 % Confidence interval plot (Mean & Median)

    Right column:
        Statistics table (AD test, descriptive stats, CIs)

    Parameters
    ----------
    data : array-like
        Numeric data vector
    column_name : str
        Variable name shown in titles and axis labels
    stats_dict : dict, optional
        Pre-computed statistics from ``eda_calculations.descriptive_statistics``.
        If *None*, statistics are computed here.
    confidence : float, default 0.95
        Confidence level for intervals
    n_bins : int, default 20
        Number of histogram bins
    height : int, default 620
    width : int, default 900

    Returns
    -------
    go.Figure
    """
    data_clean = _clean(data)

    if stats_dict is None:
        stats_dict = descriptive_statistics(data_clean)

    # ── build 3-row × 2-col figure (right col spans all rows) ──
    fig = make_subplots(
        rows=3, cols=2,
        column_widths=[0.55, 0.45],
        row_heights=[0.50, 0.20, 0.30],
        specs=[
            [{"type": "xy"},    {"type": "xy", "rowspan": 3}],
            [{"type": "xy"},    None],
            [{"type": "xy"},    None],
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    # ── 1. Histogram ─────────────────────────────────────────────
    _add_histogram(fig, data_clean, column_name, n_bins, stats_dict,
                   row=1, col=1)

    # ── 2. Boxplot ───────────────────────────────────────────────
    _add_boxplot(fig, data_clean, column_name, row=2, col=1)

    # ── 3. 95 % CI plot ──────────────────────────────────────────
    _add_ci_plot(fig, stats_dict, confidence, row=3, col=1)

    # ── 4. Statistics annotation (right column) ──────────────────
    _add_stats_panel(fig, stats_dict, column_name, confidence, row=1, col=2)

    # ── Global layout ─────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"<b>Summary Report for {column_name}</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        height=height,
        width=width,
        template='plotly_white',
        showlegend=False,
        margin=dict(l=50, r=30, t=55, b=40),
        paper_bgcolor='white',
        plot_bgcolor='white',
    )

    return fig


def plot_summary_reports_all(
    dataframe: pd.DataFrame,
    numeric_only: bool = True,
    n_bins: int = 20,
    height: int = 620,
    width: int = 900,
) -> Dict[str, go.Figure]:
    """
    Generate a Summary Report figure for every (numeric) column.

    Parameters
    ----------
    dataframe : pd.DataFrame
    numeric_only : bool, default True
    n_bins : int, default 20
    height : int, default 620
    width : int, default 900

    Returns
    -------
    dict  {column_name: go.Figure}
    """
    cols = (dataframe.select_dtypes(include=[np.number]).columns
            if numeric_only else dataframe.columns)

    figures = {}
    for col in cols:
        try:
            figures[col] = plot_summary_report(
                dataframe[col],
                column_name=col,
                n_bins=n_bins,
                height=height,
                width=width,
            )
        except Exception as e:
            figures[col] = None  # caller can check for None

    return figures


# ──────────────────────────────────────────────
#  PRIVATE HELPERS
# ──────────────────────────────────────────────

def _add_histogram(fig, data_clean, col_name, n_bins, stats_dict, row, col):
    """Histogram bars + fitted normal density curve overlay."""
    # Histogram
    fig.add_trace(go.Histogram(
        x=data_clean,
        nbinsx=n_bins,
        marker_color=_HIST_BAR,
        marker_line=dict(color=_HIST_LINE, width=0.5),
        name='Data',
        histnorm='probability density',
        hovertemplate='Count: %{y:.4f}<extra></extra>',
    ), row=row, col=col)

    # Fitted normal curve
    x_fit = np.linspace(data_clean.min(), data_clean.max(), 300)
    y_fit = sp_stats.norm.pdf(x_fit, stats_dict['mean'], stats_dict['stdev'])
    fig.add_trace(go.Scatter(
        x=x_fit,
        y=y_fit,
        mode='lines',
        line=dict(color=_NORM_CURVE, width=2.5),
        name='Normal fit',
    ), row=row, col=col)

    fig.update_xaxes(title_text=col_name, row=row, col=col,
                     tickfont=dict(size=10))
    fig.update_yaxes(title_text='Density',  row=row, col=col,
                     tickfont=dict(size=10))


def _add_boxplot(fig, data_clean, col_name, row, col):
    """Horizontal boxplot with mean diamond."""
    fig.add_trace(go.Box(
        x=data_clean,
        orientation='h',
        marker_color=_BOX_COLOR,
        line_color='#1f4e79',
        boxmean=True,           # shows mean as dashed line / diamond
        name=col_name,
        hovertemplate=(
            'Min: %{lowerfence:.3f}<br>'
            'Q1: %{q1:.3f}<br>'
            'Median: %{median:.3f}<br>'
            'Q3: %{q3:.3f}<br>'
            'Max: %{upperfence:.3f}<extra></extra>'
        ),
    ), row=row, col=col)

    fig.update_xaxes(title_text=col_name, row=row, col=col,
                     tickfont=dict(size=10))
    fig.update_yaxes(showticklabels=False, row=row, col=col)


def _add_ci_plot(fig, stats_dict, confidence, row, col):
    """Horizontal CI bars for Mean and Median (Minitab-style)."""
    ci_conf_pct = int(confidence * 100)

    entries = [
        ('Mean',   stats_dict['ci_mean'],   stats_dict['mean']),
        ('Median', stats_dict['ci_median'], stats_dict['median']),
    ]

    y_positions = [1, 0]   # Mean on top, Median below

    for y_pos, (label, (lo, hi), point) in zip(y_positions, entries):
        # CI line
        fig.add_trace(go.Scatter(
            x=[lo, hi],
            y=[y_pos, y_pos],
            mode='lines',
            line=dict(color=_CI_LINE, width=2.5),
            name=label,
            hovertemplate=f'{label} CI: [{lo:.4f}, {hi:.4f}]<extra></extra>',
        ), row=row, col=col)

        # End caps (tick marks)
        for x_cap in [lo, hi]:
            fig.add_trace(go.Scatter(
                x=[x_cap, x_cap],
                y=[y_pos - 0.15, y_pos + 0.15],
                mode='lines',
                line=dict(color=_CI_LINE, width=2.5),
                hoverinfo='skip',
                showlegend=False,
            ), row=row, col=col)

        # Point estimate (dot)
        fig.add_trace(go.Scatter(
            x=[point],
            y=[y_pos],
            mode='markers',
            marker=dict(symbol='circle', size=10, color=_CI_DOT),
            name=f'{label} estimate',
            hovertemplate=f'{label}: {point:.4f}<extra></extra>',
        ), row=row, col=col)

    fig.update_xaxes(
        title_text=f'{ci_conf_pct}% Confidence Intervals',
        row=row, col=col,
        tickfont=dict(size=10),
    )
    fig.update_yaxes(
        tickvals=[0, 1],
        ticktext=['Median', 'Mean'],
        range=[-0.6, 1.6],
        row=row, col=col,
        tickfont=dict(size=10),
    )


def _add_stats_panel(fig, s, col_name, confidence, row, col):
    """
    Invisible scatter trace + annotations to render a statistics table
    in the right column, matching Minitab's layout.
    """
    ci_pct = int(confidence * 100)

    # ── build text lines ────────────────────────────────────────
    ad_color  = 'red' if s['ad_reject'] else '#1a6b1a'
    ad_norm   = 'Not normal' if s['ad_reject'] else 'Normal'

    lines: List[Dict] = [
        # Section: AD test
        {'text': f"<b>Anderson-Darling Normality Test</b>",
         'bold': True, 'offset': 0.00},
        {'text': f"    A-Squared        {s['ad_statistic']:.4f}",
         'offset': -0.06},
        {'text': f"    P-Value          {s['ad_p_label']}",
         'offset': -0.12, 'color': ad_color},
        {'text': f"    Conclusion:  {ad_norm}",
         'offset': -0.18, 'color': ad_color},

        # Blank spacer
        {'text': '', 'offset': -0.24},

        # Section: Descriptive
        {'text': f"<b>Descriptive Statistics</b>",
         'bold': True, 'offset': -0.26},
        {'text': f"    Mean            {s['mean']:.4f}",    'offset': -0.32},
        {'text': f"    StDev           {s['stdev']:.4f}",   'offset': -0.38},
        {'text': f"    Variance        {s['variance']:.4f}",'offset': -0.44},
        {'text': f"    Skewness        {s['skewness']:.6f}",'offset': -0.50},
        {'text': f"    Kurtosis        {s['kurtosis']:.6f}",'offset': -0.56},
        {'text': f"    N               {s['n']}",           'offset': -0.62},

        # Blank spacer
        {'text': '', 'offset': -0.66},

        # Section: 5-number summary
        {'text': f"<b>5-Number Summary</b>",
         'bold': True, 'offset': -0.68},
        {'text': f"    Minimum         {s['minimum']:.4f}", 'offset': -0.74},
        {'text': f"    1st Quartile    {s['q1']:.4f}",      'offset': -0.80},
        {'text': f"    Median          {s['median']:.4f}",  'offset': -0.86},
        {'text': f"    3rd Quartile    {s['q3']:.4f}",      'offset': -0.92},
        {'text': f"    Maximum         {s['maximum']:.4f}", 'offset': -0.98},

        # Blank spacer
        {'text': '', 'offset': -1.02},

        # Section: Confidence intervals
        {'text': f"<b>{ci_pct}% Confidence Intervals</b>",
         'bold': True, 'offset': -1.04},
        {'text': f"    Mean      [{s['ci_mean'][0]:.3f},  {s['ci_mean'][1]:.3f}]",
         'offset': -1.10},
        {'text': f"    Median    [{s['ci_median'][0]:.3f},  {s['ci_median'][1]:.3f}]",
         'offset': -1.16},
        {'text': f"    StDev     [{s['ci_stdev'][0]:.3f},  {s['ci_stdev'][1]:.3f}]",
         'offset': -1.22},
    ]

    # Dummy invisible trace to anchor the annotation area
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(opacity=0),
        showlegend=False,
        hoverinfo='skip',
    ), row=row, col=col)

    # Hide axes in stats panel
    fig.update_xaxes(visible=False, row=row, col=col)
    fig.update_yaxes(visible=False, row=row, col=col)

    # Anchor annotations to the subplot's paper domain.
    # We use xref/yref relative to the figure to place text freely.
    # The right subplot occupies roughly x=[0.61, 0.98] in paper coords.
    x_anchor = 0.595   # left edge of right panel (paper fraction)
    y_start  = 0.970   # top of right panel (paper fraction)

    line_height = 0.048  # vertical step per line in paper fraction

    annotations = []
    for i, line in enumerate(lines):
        if not line.get('text'):
            continue
        color = line.get('color', '#222222')
        annotations.append(dict(
            text=f"<span style='font-family:Courier New,monospace; font-size:11px; color:{color}'>{line['text']}</span>",
            x=x_anchor,
            y=y_start + line['offset'] * (line_height / 0.06),
            xref='paper',
            yref='paper',
            showarrow=False,
            xanchor='left',
            yanchor='top',
            align='left',
        ))

    fig.update_layout(annotations=annotations)
