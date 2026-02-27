"""
Two-Sample T-Test Plots Module

Minitab-style visualizations for two-sample t-test:
- Individual Value Plot (with group means and CIs)
- Comparative Boxplot
- Histogram overlay by group
- Normal Q-Q Plot by group
- Interval Plot (mean ± CI)
- Effect size forest plot
- Test summary report card

Integrates with color_utils.py when available.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from typing import Dict, Any, Optional, List, Union


# ===== COLOR DEFAULTS =====
try:
    from color_utils import get_unified_color_schemes, create_categorical_color_map
    _scheme = get_unified_color_schemes()
    DEFAULT_COLORS = _scheme['categorical_colors'][:2]  # first two categorical colors
except ImportError:
    DEFAULT_COLORS = ['steelblue', 'coral']


def _get_colors(label1: str, label2: str):
    """Get consistent colors for two groups."""
    try:
        cmap = create_categorical_color_map([label1, label2])
        return cmap[label1], cmap[label2]
    except Exception:
        return DEFAULT_COLORS[0], DEFAULT_COLORS[1]


# ========== 1. INDIVIDUAL VALUE PLOT (MINITAB SIGNATURE) ==========

def plot_individual_value(
    sample1: np.ndarray,
    sample2: np.ndarray,
    label1: str = "Sample 1",
    label2: str = "Sample 2",
    results: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None
) -> go.Figure:
    """
    Minitab-style Individual Value Plot.

    Shows all data points with group means and confidence intervals.

    Parameters
    ----------
    sample1, sample2 : array-like
    label1, label2 : str
    results : dict, optional
        Output from two_sample_ttest() to overlay statistics
    title : str, optional

    Returns
    -------
    go.Figure
    """
    s1 = np.asarray(sample1, dtype=float).flatten()
    s2 = np.asarray(sample2, dtype=float).flatten()
    s1, s2 = s1[~np.isnan(s1)], s2[~np.isnan(s2)]

    color1, color2 = _get_colors(label1, label2)

    if title is None:
        title = "Individual Value Plot"

    fig = go.Figure()

    # Jittered x positions
    jitter1 = np.random.normal(0, 0.03, len(s1))
    jitter2 = np.random.normal(1, 0.03, len(s2))

    # Data points
    fig.add_trace(go.Scatter(
        x=jitter1, y=s1,
        mode='markers',
        name=label1,
        marker=dict(color=color1, size=7, opacity=0.65,
                    line=dict(width=0.5, color='white')),
        hovertemplate=f"<b>{label1}</b><br>Value: %{{y:.4f}}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=jitter2, y=s2,
        mode='markers',
        name=label2,
        marker=dict(color=color2, size=7, opacity=0.65,
                    line=dict(width=0.5, color='white')),
        hovertemplate=f"<b>{label2}</b><br>Value: %{{y:.4f}}<extra></extra>"
    ))

    # Group means (larger marker + horizontal line)
    mean1, mean2 = np.mean(s1), np.mean(s2)

    fig.add_trace(go.Scatter(
        x=[0], y=[mean1],
        mode='markers',
        marker=dict(symbol='diamond', size=14, color=color1,
                    line=dict(width=2, color='black')),
        name=f"Mean {label1}: {mean1:.4f}",
        showlegend=True,
        hovertemplate=f"<b>Mean {label1}</b>: %{{y:.4f}}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=[1], y=[mean2],
        mode='markers',
        marker=dict(symbol='diamond', size=14, color=color2,
                    line=dict(width=2, color='black')),
        name=f"Mean {label2}: {mean2:.4f}",
        showlegend=True,
        hovertemplate=f"<b>Mean {label2}</b>: %{{y:.4f}}<extra></extra>"
    ))

    # Mean lines
    fig.add_shape(type="line", x0=-0.25, x1=0.25, y0=mean1, y1=mean1,
                  line=dict(color=color1, width=2, dash='dash'))
    fig.add_shape(type="line", x0=0.75, x1=1.25, y0=mean2, y1=mean2,
                  line=dict(color=color2, width=2, dash='dash'))

    # CI bars if results provided
    if results:
        cl = results.get('ci', {}).get('confidence_level', 0.95)
        alpha = 1 - cl
        se1 = results['descriptive']['sample1']['se_mean']
        se2 = results['descriptive']['sample2']['se_mean']
        n1 = results['descriptive']['sample1']['n']
        n2 = results['descriptive']['sample2']['n']
        t_crit1 = stats.t.ppf(1 - alpha/2, n1 - 1)
        t_crit2 = stats.t.ppf(1 - alpha/2, n2 - 1)

        # CI for sample 1
        ci1_lo = mean1 - t_crit1 * se1
        ci1_hi = mean1 + t_crit1 * se1
        fig.add_shape(type="line", x0=0, x1=0, y0=ci1_lo, y1=ci1_hi,
                      line=dict(color='black', width=2))
        fig.add_shape(type="line", x0=-0.05, x1=0.05, y0=ci1_lo, y1=ci1_lo,
                      line=dict(color='black', width=2))
        fig.add_shape(type="line", x0=-0.05, x1=0.05, y0=ci1_hi, y1=ci1_hi,
                      line=dict(color='black', width=2))

        # CI for sample 2
        ci2_lo = mean2 - t_crit2 * se2
        ci2_hi = mean2 + t_crit2 * se2
        fig.add_shape(type="line", x0=1, x1=1, y0=ci2_lo, y1=ci2_hi,
                      line=dict(color='black', width=2))
        fig.add_shape(type="line", x0=0.95, x1=1.05, y0=ci2_lo, y1=ci2_lo,
                      line=dict(color='black', width=2))
        fig.add_shape(type="line", x0=0.95, x1=1.05, y0=ci2_hi, y1=ci2_hi,
                      line=dict(color='black', width=2))

    fig.update_layout(
        title=title,
        xaxis=dict(
            tickvals=[0, 1],
            ticktext=[label1, label2],
            title="Group",
            range=[-0.5, 1.5]
        ),
        yaxis_title="Value",
        template='plotly_white',
        hovermode='closest',
        height=500
    )

    return fig


# ========== 2. COMPARATIVE BOXPLOT ==========

def plot_comparative_boxplot(
    sample1: np.ndarray,
    sample2: np.ndarray,
    label1: str = "Sample 1",
    label2: str = "Sample 2",
    title: Optional[str] = None
) -> go.Figure:
    """
    Side-by-side boxplots (Minitab-style) with mean markers.
    """
    s1 = np.asarray(sample1, dtype=float).flatten()
    s2 = np.asarray(sample2, dtype=float).flatten()
    s1, s2 = s1[~np.isnan(s1)], s2[~np.isnan(s2)]

    color1, color2 = _get_colors(label1, label2)

    if title is None:
        title = "Boxplot of Data"

    fig = go.Figure()

    fig.add_trace(go.Box(
        y=s1,
        name=label1,
        marker_color=color1,
        boxmean='sd',
        boxpoints='outliers',
        jitter=0.3,
        pointpos=-1.5,
        hovertemplate=f"<b>{label1}</b><br>%{{y:.4f}}<extra></extra>"
    ))

    fig.add_trace(go.Box(
        y=s2,
        name=label2,
        marker_color=color2,
        boxmean='sd',
        boxpoints='outliers',
        jitter=0.3,
        pointpos=-1.5,
        hovertemplate=f"<b>{label2}</b><br>%{{y:.4f}}<extra></extra>"
    ))

    fig.update_layout(
        title=title,
        yaxis_title="Value",
        xaxis_title="Group",
        template='plotly_white',
        height=500,
        showlegend=False
    )

    return fig


# ========== 3. HISTOGRAM OVERLAY ==========

def plot_histogram_overlay(
    sample1: np.ndarray,
    sample2: np.ndarray,
    label1: str = "Sample 1",
    label2: str = "Sample 2",
    n_bins: int = 20,
    title: Optional[str] = None
) -> go.Figure:
    """
    Overlaid histograms with mean lines for both groups.
    """
    s1 = np.asarray(sample1, dtype=float).flatten()
    s2 = np.asarray(sample2, dtype=float).flatten()
    s1, s2 = s1[~np.isnan(s1)], s2[~np.isnan(s2)]

    color1, color2 = _get_colors(label1, label2)

    if title is None:
        title = "Histogram of Data"

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=s1, name=label1,
        marker_color=color1, opacity=0.6,
        nbinsx=n_bins
    ))

    fig.add_trace(go.Histogram(
        x=s2, name=label2,
        marker_color=color2, opacity=0.6,
        nbinsx=n_bins
    ))

    # Mean lines
    fig.add_vline(x=np.mean(s1), line_dash="dash", line_color=color1,
                  annotation_text=f"Mean {label1}: {np.mean(s1):.3f}",
                  annotation_position="top left")
    fig.add_vline(x=np.mean(s2), line_dash="dash", line_color=color2,
                  annotation_text=f"Mean {label2}: {np.mean(s2):.3f}",
                  annotation_position="top right")

    fig.update_layout(
        barmode='overlay',
        title=title,
        xaxis_title="Value",
        yaxis_title="Frequency",
        template='plotly_white',
        height=450
    )

    return fig


# ========== 4. NORMAL Q-Q PLOT BY GROUP ==========

def plot_qq_by_group(
    sample1: np.ndarray,
    sample2: np.ndarray,
    label1: str = "Sample 1",
    label2: str = "Sample 2",
    title: Optional[str] = None
) -> go.Figure:
    """
    Q-Q plots for both groups side by side (normality assessment).
    """
    s1 = np.asarray(sample1, dtype=float).flatten()
    s2 = np.asarray(sample2, dtype=float).flatten()
    s1, s2 = s1[~np.isnan(s1)], s2[~np.isnan(s2)]

    color1, color2 = _get_colors(label1, label2)

    if title is None:
        title = "Normal Q-Q Plot"

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Q-Q Plot: {label1}", f"Q-Q Plot: {label2}")
    )

    for idx, (s, label, color) in enumerate([(s1, label1, color1), (s2, label2, color2)]):
        sorted_data = np.sort(s)
        n = len(sorted_data)
        theoretical = stats.norm.ppf((np.arange(1, n+1) - 0.375) / (n + 0.25))

        fig.add_trace(
            go.Scatter(
                x=theoretical, y=sorted_data,
                mode='markers',
                name=label,
                marker=dict(color=color, size=6, opacity=0.7),
                hovertemplate="Theoretical: %{x:.3f}<br>Observed: %{y:.4f}<extra></extra>"
            ),
            row=1, col=idx+1
        )

        # Reference line (fit to data)
        slope, intercept = np.polyfit(theoretical, sorted_data, 1)
        fit_line = slope * theoretical + intercept
        fig.add_trace(
            go.Scatter(
                x=theoretical, y=fit_line,
                mode='lines',
                line=dict(color='red', dash='dash', width=1.5),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=idx+1
        )

        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=idx+1)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=idx+1)

    fig.update_layout(
        title=title,
        template='plotly_white',
        height=450
    )

    return fig


# ========== 5. INTERVAL PLOT (MEAN ± CI) ==========

def plot_interval_plot(
    sample1: np.ndarray,
    sample2: np.ndarray,
    label1: str = "Sample 1",
    label2: str = "Sample 2",
    confidence_level: float = 0.95,
    title: Optional[str] = None
) -> go.Figure:
    """
    Interval plot showing mean ± CI for each group and for the difference.
    """
    s1 = np.asarray(sample1, dtype=float).flatten()
    s2 = np.asarray(sample2, dtype=float).flatten()
    s1, s2 = s1[~np.isnan(s1)], s2[~np.isnan(s2)]

    color1, color2 = _get_colors(label1, label2)
    alpha = 1 - confidence_level
    ci_pct = int(confidence_level * 100)

    if title is None:
        title = f"Interval Plot ({ci_pct}% CI for the Mean)"

    # Calculate CIs
    mean1, se1 = np.mean(s1), stats.sem(s1)
    mean2, se2 = np.mean(s2), stats.sem(s2)
    t_crit1 = stats.t.ppf(1 - alpha/2, len(s1) - 1)
    t_crit2 = stats.t.ppf(1 - alpha/2, len(s2) - 1)

    fig = go.Figure()

    # Group 1
    fig.add_trace(go.Scatter(
        x=[label1], y=[mean1],
        mode='markers',
        marker=dict(symbol='circle', size=12, color=color1,
                    line=dict(width=1, color='black')),
        error_y=dict(
            type='constant',
            value=t_crit1 * se1,
            color=color1,
            thickness=2,
            width=8
        ),
        name=f"{label1}: {mean1:.4f}",
        hovertemplate=(
            f"<b>{label1}</b><br>"
            f"Mean: %{{y:.4f}}<br>"
            f"CI: ({mean1 - t_crit1*se1:.4f}, {mean1 + t_crit1*se1:.4f})"
            f"<extra></extra>"
        )
    ))

    # Group 2
    fig.add_trace(go.Scatter(
        x=[label2], y=[mean2],
        mode='markers',
        marker=dict(symbol='circle', size=12, color=color2,
                    line=dict(width=1, color='black')),
        error_y=dict(
            type='constant',
            value=t_crit2 * se2,
            color=color2,
            thickness=2,
            width=8
        ),
        name=f"{label2}: {mean2:.4f}",
        hovertemplate=(
            f"<b>{label2}</b><br>"
            f"Mean: %{{y:.4f}}<br>"
            f"CI: ({mean2 - t_crit2*se2:.4f}, {mean2 + t_crit2*se2:.4f})"
            f"<extra></extra>"
        )
    ))

    # Pooled mean reference line
    pooled_mean = (np.sum(s1) + np.sum(s2)) / (len(s1) + len(s2))
    fig.add_hline(y=pooled_mean, line_dash="dot", line_color="gray",
                  annotation_text=f"Pooled mean: {pooled_mean:.4f}",
                  annotation_position="top right")

    fig.update_layout(
        title=title,
        yaxis_title="Value",
        xaxis_title="Group",
        template='plotly_white',
        height=450,
        showlegend=True
    )

    return fig


# ========== 6. FOUR-IN-ONE DIAGNOSTIC (MINITAB EDA) ==========

def plot_ttest_fourplot(
    sample1: np.ndarray,
    sample2: np.ndarray,
    label1: str = "Sample 1",
    label2: str = "Sample 2",
    results: Optional[Dict[str, Any]] = None,
    confidence_level: float = 0.95
) -> go.Figure:
    """
    Minitab-style 4-in-1 summary plot:
    1. Individual Value Plot
    2. Boxplot
    3. Histogram overlay
    4. Q-Q plots

    Parameters
    ----------
    sample1, sample2 : array-like
    label1, label2 : str
    results : dict, optional
    confidence_level : float

    Returns
    -------
    go.Figure
    """
    s1 = np.asarray(sample1, dtype=float).flatten()
    s2 = np.asarray(sample2, dtype=float).flatten()
    s1, s2 = s1[~np.isnan(s1)], s2[~np.isnan(s2)]

    color1, color2 = _get_colors(label1, label2)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Individual Value Plot",
            "Boxplot",
            "Histogram",
            "Normal Q-Q Plot"
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.12
    )

    # 1. Individual Value Plot (top-left)
    jitter1 = np.random.normal(0, 0.04, len(s1))
    jitter2 = np.random.normal(1, 0.04, len(s2))

    fig.add_trace(go.Scatter(
        x=jitter1, y=s1, mode='markers', name=label1,
        marker=dict(color=color1, size=5, opacity=0.6),
        legendgroup=label1, showlegend=True
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=jitter2, y=s2, mode='markers', name=label2,
        marker=dict(color=color2, size=5, opacity=0.6),
        legendgroup=label2, showlegend=True
    ), row=1, col=1)

    fig.update_xaxes(tickvals=[0, 1], ticktext=[label1, label2], row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)

    # 2. Boxplot (top-right)
    fig.add_trace(go.Box(
        y=s1, name=label1, marker_color=color1, boxmean='sd',
        legendgroup=label1, showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Box(
        y=s2, name=label2, marker_color=color2, boxmean='sd',
        legendgroup=label2, showlegend=False
    ), row=1, col=2)

    fig.update_yaxes(title_text="Value", row=1, col=2)

    # 3. Histogram (bottom-left)
    fig.add_trace(go.Histogram(
        x=s1, name=label1, marker_color=color1, opacity=0.6,
        nbinsx=15, legendgroup=label1, showlegend=False
    ), row=2, col=1)

    fig.add_trace(go.Histogram(
        x=s2, name=label2, marker_color=color2, opacity=0.6,
        nbinsx=15, legendgroup=label2, showlegend=False
    ), row=2, col=1)

    fig.update_xaxes(title_text="Value", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)

    # 4. Q-Q Plot (bottom-right) - both groups
    for s, label, color in [(s1, label1, color1), (s2, label2, color2)]:
        sorted_data = np.sort(s)
        n = len(sorted_data)
        theoretical = stats.norm.ppf((np.arange(1, n+1) - 0.375) / (n + 0.25))

        fig.add_trace(go.Scatter(
            x=theoretical, y=sorted_data, mode='markers',
            name=label, marker=dict(color=color, size=4, opacity=0.7),
            legendgroup=label, showlegend=False
        ), row=2, col=2)

    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)

    # Annotation with test results
    if results:
        test = results['test_results']
        ann_text = (
            f"t = {test['t_statistic']:.3f}, "
            f"df = {test['df']:.1f}, "
            f"p = {test['p_value']:.4f}"
        )
        fig.add_annotation(
            text=ann_text,
            xref="paper", yref="paper",
            x=0.5, y=1.06,
            showarrow=False,
            font=dict(size=12, color="darkred"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="darkred",
            borderwidth=1,
            borderpad=4
        )

    fig.update_layout(
        title="Two-Sample T-Test: Summary Plots",
        height=700,
        template='plotly_white',
        barmode='overlay',
        hovermode='closest'
    )

    return fig


# ========== 7. TEST RESULT REPORT FIGURE ==========

def plot_test_report(
    results: Dict[str, Any],
    label1: str = "Sample 1",
    label2: str = "Sample 2"
) -> go.Figure:
    """
    Visual test report card showing key results.

    Creates a clean figure with test statistics, CI, and effect size
    displayed as annotations (similar to Minitab's report card).
    """
    test = results['test_results']
    desc = results['descriptive']
    ci = results['ci']
    eff = results['effect_size']
    ci_pct = int(ci['confidence_level'] * 100)

    fig = go.Figure()

    # Determine significance
    alpha = 1 - ci['confidence_level']
    significant = test['p_value'] < alpha
    verdict_color = "green" if significant else "gray"
    verdict_text = "Significant" if significant else "Not Significant"

    # Difference with CI visualization
    diff = test['difference']
    ci_lo = ci['lower'] if np.isfinite(ci['lower']) else diff - 3 * test['se_difference']
    ci_hi = ci['upper'] if np.isfinite(ci['upper']) else diff + 3 * test['se_difference']

    # CI bar
    fig.add_trace(go.Scatter(
        x=[ci_lo, ci_hi], y=[0.5, 0.5],
        mode='lines',
        line=dict(color=verdict_color, width=4),
        showlegend=False,
        hoverinfo='skip'
    ))

    # CI endpoints
    fig.add_trace(go.Scatter(
        x=[ci_lo, ci_hi], y=[0.5, 0.5],
        mode='markers',
        marker=dict(symbol='line-ns', size=15, color=verdict_color, line_width=3),
        showlegend=False,
        hovertemplate=f"{ci_pct}% CI: (%{{x:.4f}})<extra></extra>"
    ))

    # Difference point
    fig.add_trace(go.Scatter(
        x=[diff], y=[0.5],
        mode='markers',
        marker=dict(symbol='diamond', size=16, color='black',
                    line=dict(width=2, color='white')),
        name=f"Difference = {diff:.4f}",
        hovertemplate=f"<b>Difference</b>: {diff:.4f}<extra></extra>"
    ))

    # Reference line at hypothesized difference
    hyp = test['hypothesized_difference']
    fig.add_vline(x=hyp, line_dash="dash", line_color="red", line_width=2)
    fig.add_annotation(
        x=hyp, y=0.7,
        text=f"H₀: Δ = {hyp}",
        showarrow=False,
        font=dict(color="red", size=11)
    )

    # Add text summary
    text_lines = [
        f"<b>Two-Sample T-Test Results</b>",
        f"",
        f"<b>{label1}</b>: n={desc['sample1']['n']}, "
        f"mean={desc['sample1']['mean']:.4f}, "
        f"StDev={desc['sample1']['std_dev']:.4f}",
        f"<b>{label2}</b>: n={desc['sample2']['n']}, "
        f"mean={desc['sample2']['mean']:.4f}, "
        f"StDev={desc['sample2']['std_dev']:.4f}",
        f"",
        f"Difference ({label1} − {label2}): <b>{diff:.4f}</b>",
        f"{ci_pct}% CI: ({ci_lo:.4f}, {ci_hi:.4f})",
        f"",
        f"T-Value = {test['t_statistic']:.4f}    "
        f"DF = {test['df']:.2f}    "
        f"P-Value = {test['p_value']:.6f}",
        f"",
        f"Cohen's d = {eff['cohens_d']:.4f}    "
        f"Verdict: <b>{verdict_text}</b>"
    ]

    fig.add_annotation(
        text="<br>".join(text_lines),
        xref="paper", yref="paper",
        x=0.5, y=1.25,
        showarrow=False,
        font=dict(size=12, family="monospace"),
        align="left",
        bgcolor="rgba(245, 245, 245, 0.95)",
        bordercolor="gray",
        borderwidth=1,
        borderpad=10
    )

    fig.update_layout(
        title="",
        xaxis_title=f"Difference of Means ({label1} − {label2})",
        yaxis=dict(visible=False, range=[0, 1]),
        template='plotly_white',
        height=350,
        margin=dict(t=220),
        showlegend=False
    )

    return fig
