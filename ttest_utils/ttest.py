"""
Two-Sample t-Test – Full Minitab-Style Implementation + Sample Size Calculator
+ Before/After Capability Comparison (Minitab-style)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, f as f_dist
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional, Tuple


# =============================================================================
# EXISTING FUNCTIONS (unchanged)
# =============================================================================

def perform_two_sample_ttest(
    data: pd.DataFrame,
    response_col: Optional[str] = None,
    group_col: Optional[str] = None,
    col1: Optional[str] = None,
    col2: Optional[str] = None,
    equal_var: bool = True,
    conf_level: float = 0.95,
    alternative: str = "two-sided"
) -> Dict:
    """Performs 2-sample t-test exactly as shown in Minitab."""

    if col1 and col2:  # Unstacked
        sample1 = pd.to_numeric(data[col1], errors='coerce').dropna()
        sample2 = pd.to_numeric(data[col2], errors='coerce').dropna()
        name1, name2 = col1, col2
    else:  # Stacked
        groups = sorted(data[group_col].dropna().unique())
        if len(groups) != 2:
            raise ValueError(f"Grouping column must contain exactly 2 levels")
        name1, name2 = groups
        sample1 = pd.to_numeric(data[data[group_col] == name1][response_col], errors='coerce').dropna()
        sample2 = pd.to_numeric(data[data[group_col] == name2][response_col], errors='coerce').dropna()

    result = stats.ttest_ind(sample1, sample2, equal_var=equal_var, alternative=alternative, nan_policy='omit')
    ci = result.confidence_interval(confidence_level=conf_level)

    def desc(s: pd.Series) -> Dict:
        return {
            'N': len(s),
            'Mean': s.mean(),
            'StDev': s.std(ddof=1),
            'SE Mean': s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 0 else np.nan
        }

    return {
        'groups': [name1, name2],
        'descriptive': {name1: desc(sample1), name2: desc(sample2)},
        'difference': {
            'estimate': sample1.mean() - sample2.mean(),
            'ci_lower': ci.low,
            'ci_upper': ci.high,
            'confidence': conf_level * 100
        },
        'test': {
            't_value': result.statistic,
            'df': result.df,
            'p_value': result.pvalue,
            'equal_var_assumed': equal_var,
            'alternative': alternative
        },
        'conclusion': "Reject H₀ – Significant difference" if result.pvalue < (1 - conf_level) else "Fail to reject H₀ – No significant difference"
    }


def plot_minitab_style_comparison(
    data: pd.DataFrame,
    response_col: str,
    group_col: Optional[str] = None
) -> go.Figure:
    fig = go.Figure()
    if group_col:
        groups = sorted(data[group_col].dropna().unique())
        for g in groups:
            values = data[data[group_col] == g][response_col]
            fig.add_trace(go.Box(y=values, name=str(g), boxpoints='all', jitter=0.3, pointpos=-1.8, marker=dict(size=6)))
    else:
        fig.add_trace(go.Box(y=data[response_col], name=response_col))

    fig.update_layout(title="2-Sample Comparison (Minitab Style)", yaxis_title=response_col, template="plotly_white", height=520, showlegend=True)
    return fig


def calculate_sample_size_tost(sigma: float, delta: float, alpha: float = 0.05, power: float = 0.80) -> Dict:
    """Schuirmann's TOST sample size calculation."""
    z_alpha = norm.ppf(1 - alpha)
    z_beta = norm.ppf(1 - power / 2)
    n_per_group = 2 * ((sigma / delta) ** 2) * (z_alpha + z_beta) ** 2
    n_per_group = int(np.ceil(n_per_group))
    return {'n_per_group': n_per_group, 'total_samples': n_per_group * 2, 'sigma': sigma, 'delta': delta, 'alpha': alpha, 'power': power}


def estimate_pooled_sd(data: pd.DataFrame, response_col: str, group_col: str) -> float:
    """Auto-estimate pooled standard deviation from dataset."""
    groups = data[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError("Grouping variable must have exactly 2 levels")

    group1 = pd.to_numeric(data[data[group_col] == groups[0]][response_col], errors='coerce').dropna()
    group2 = pd.to_numeric(data[data[group_col] == groups[1]][response_col], errors='coerce').dropna()

    n1, n2 = len(group1), len(group2)
    s1, s2 = group1.std(ddof=1), group2.std(ddof=1)
    pooled_var = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
    return np.sqrt(pooled_var)


# =============================================================================
# CAPABILITY ANALYSIS – Minitab Before/After Comparison
# =============================================================================

def perform_capability_analysis(
    before: np.ndarray,
    after: np.ndarray,
    usl: Optional[float] = None,
    lsl: Optional[float] = None,
    target: Optional[float] = None,
    alpha: float = 0.05
) -> Dict:
    """
    Full Minitab-style Before / After Capability Comparison.

    Returns a dict with every metric shown in the Minitab summary report:
    process characterization, capability indices, hypothesis-test results,
    reduction statistics and auto-generated comments.
    """
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    before = before[~np.isnan(before)]
    after = after[~np.isnan(after)]

    if len(before) < 2 or len(after) < 2:
        raise ValueError("Both samples need at least 2 observations.")
    if usl is None and lsl is None:
        raise ValueError("At least one specification limit (USL or LSL) is required.")

    # ---- descriptive --------------------------------------------------------
    b_mean, a_mean = before.mean(), after.mean()
    b_std, a_std = before.std(ddof=1), after.std(ddof=1)
    b_n, a_n = len(before), len(after)

    # ---- capability indices -------------------------------------------------
    def _capability(mean, std, usl, lsl):
        pp = ppl = ppu = ppk = None
        if usl is not None and lsl is not None:
            pp = (usl - lsl) / (6.0 * std)
            ppu = (usl - mean) / (3.0 * std)
            ppl = (mean - lsl) / (3.0 * std)
            ppk = min(ppu, ppl)
        elif usl is not None:
            ppu = (usl - mean) / (3.0 * std)
            ppk = ppu
        elif lsl is not None:
            ppl = (mean - lsl) / (3.0 * std)
            ppk = ppl
        return pp, ppk

    b_pp, b_ppk = _capability(b_mean, b_std, usl, lsl)
    a_pp, a_ppk = _capability(a_mean, a_std, usl, lsl)

    # ---- % out of spec & PPM ------------------------------------------------
    def _out_of_spec(mean, std, usl, lsl):
        prob = 0.0
        if usl is not None:
            prob += 1.0 - norm.cdf(usl, loc=mean, scale=std)
        if lsl is not None:
            prob += norm.cdf(lsl, loc=mean, scale=std)
        return prob

    b_prob = _out_of_spec(b_mean, b_std, usl, lsl)
    a_prob = _out_of_spec(a_mean, a_std, usl, lsl)

    b_pct = b_prob * 100.0
    a_pct = a_prob * 100.0
    b_ppm = b_prob * 1_000_000.0
    a_ppm = a_prob * 1_000_000.0

    # ---- Z.Bench ------------------------------------------------------------
    def _zbench(prob_out):
        prob_out = np.clip(prob_out, 1e-15, 1.0 - 1e-15)
        return norm.ppf(1.0 - prob_out)

    b_zbench = _zbench(b_prob)
    a_zbench = _zbench(a_prob)

    # ---- Reduction ----------------------------------------------------------
    if b_pct > 0:
        reduction_pct = ((b_pct - a_pct) / b_pct) * 100.0
    elif a_pct == 0:
        reduction_pct = 0.0
    else:
        reduction_pct = -100.0  # got worse from zero

    # ---- hypothesis tests ---------------------------------------------------
    # F-test: was SD reduced? (one-sided: H1 = before_var > after_var)
    f_stat = b_std**2 / a_std**2
    df1, df2 = b_n - 1, a_n - 1
    f_p_value = 1.0 - f_dist.cdf(f_stat, df1, df2)       # one-sided upper
    sd_reduced = f_p_value < alpha

    # Two-sample t-test: did mean change? (two-sided, Welch)
    t_result = stats.ttest_ind(before, after, equal_var=False)
    t_p_value = t_result.pvalue
    mean_changed = t_p_value < alpha

    # ---- comments -----------------------------------------------------------
    comments = []
    if sd_reduced:
        comments.append(f"The process standard deviation was reduced significantly (p < {alpha}).")
    else:
        comments.append(f"The process standard deviation was NOT reduced significantly (p > {alpha}).")
    if mean_changed:
        comments.append(f"The process mean changed significantly (p < {alpha}).")
    else:
        comments.append(f"The process mean did not change significantly (p > {alpha}).")
    comments.append("")
    comments.append("Actual (overall) capability is what the customer experiences.")
    comments.append("Potential (within) capability is what could be achieved if process shifts and drifts were eliminated.")

    return {
        'before': {'mean': b_mean, 'std': b_std, 'n': b_n, 'data': before},
        'after':  {'mean': a_mean, 'std': a_std,  'n': a_n, 'data': after},
        'change': {'mean': a_mean - b_mean, 'std': a_std - b_std},
        'capability': {
            'before_pp': b_pp, 'after_pp': a_pp,
            'change_pp': (a_pp - b_pp) if (a_pp is not None and b_pp is not None) else None,
            'before_ppk': b_ppk, 'after_ppk': a_ppk,
            'change_ppk': (a_ppk - b_ppk) if (a_ppk is not None and b_ppk is not None) else None,
        },
        'zbench':      {'before': b_zbench, 'after': a_zbench, 'change': a_zbench - b_zbench},
        'out_of_spec': {'before': b_pct,    'after': a_pct,    'change': a_pct - b_pct},
        'ppm':         {'before': b_ppm,    'after': a_ppm,    'change': a_ppm - b_ppm},
        'reduction_pct': reduction_pct,
        'tests': {
            'f_test': {'statistic': f_stat, 'p_value': f_p_value, 'sd_reduced': sd_reduced},
            't_test': {'statistic': t_result.statistic, 'p_value': t_p_value, 'mean_changed': mean_changed},
        },
        'specs': {'usl': usl, 'lsl': lsl, 'target': target},
        'comments': comments,
    }


def plot_capability_histograms(
    result: Dict,
    before_label: str = "Before",
    after_label: str = "After",
) -> go.Figure:
    """
    Minitab-style stacked histograms (Before on top, After on bottom)
    with normal-curve overlays and specification-limit lines.
    """
    before = result['before']['data']
    after  = result['after']['data']
    usl    = result['specs']['usl']
    lsl    = result['specs']['lsl']
    b_mean, b_std = result['before']['mean'], result['before']['std']
    a_mean, a_std = result['after']['mean'],  result['after']['std']

    # shared x-range
    all_vals = np.concatenate([before, after])
    pad = (all_vals.max() - all_vals.min()) * 0.25
    x_lo = all_vals.min() - pad
    x_hi = all_vals.max() + pad
    if usl is not None:
        x_hi = max(x_hi, usl + pad * 0.5)
    if lsl is not None:
        x_lo = min(x_lo, lsl - pad * 0.5)

    x_curve = np.linspace(x_lo, x_hi, 300)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(before_label, after_label),
        vertical_spacing=0.12,
        shared_xaxes=True,
    )

    # Determine good bin edges shared between both
    n_bins = max(8, int(np.sqrt(max(len(before), len(after)))))
    bin_edges = np.linspace(x_lo, x_hi, n_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]

    for row, (data_arr, mean, std, label) in enumerate([
        (before, b_mean, b_std, before_label),
        (after,  a_mean, a_std, after_label),
    ], start=1):
        # histogram
        counts, _ = np.histogram(data_arr, bins=bin_edges)
        # scale normal curve to match histogram area
        area = len(data_arr) * bin_width
        y_curve = norm.pdf(x_curve, loc=mean, scale=std) * area

        fig.add_trace(go.Bar(
            x=(bin_edges[:-1] + bin_edges[1:]) / 2,
            y=counts,
            width=bin_width * 0.92,
            marker_color='steelblue',
            marker_line=dict(color='white', width=0.5),
            opacity=0.85,
            name=label,
            showlegend=False,
        ), row=row, col=1)

        fig.add_trace(go.Scatter(
            x=x_curve, y=y_curve,
            mode='lines',
            line=dict(color='firebrick', width=2.5),
            name='Normal fit',
            showlegend=(row == 1),
        ), row=row, col=1)

        # spec lines
        y_max = max(counts.max() * 1.15, y_curve.max() * 1.15) if len(counts) > 0 else 1
        if usl is not None:
            fig.add_trace(go.Scatter(
                x=[usl, usl], y=[0, y_max],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='USL' if row == 1 else None,
                showlegend=(row == 1),
            ), row=row, col=1)
            fig.add_annotation(
                x=usl, y=y_max, text="USL", showarrow=False,
                font=dict(color='red', size=11, family='Arial Black'),
                xanchor='left', yanchor='bottom',
                row=row, col=1,
            )
        if lsl is not None:
            fig.add_trace(go.Scatter(
                x=[lsl, lsl], y=[0, y_max],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='LSL' if row == 1 else None,
                showlegend=(row == 1),
            ), row=row, col=1)
            fig.add_annotation(
                x=lsl, y=y_max, text="LSL", showarrow=False,
                font=dict(color='red', size=11, family='Arial Black'),
                xanchor='right', yanchor='bottom',
                row=row, col=1,
            )

    fig.update_layout(
        height=520,
        template='plotly_white',
        margin=dict(l=50, r=30, t=50, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=-0.12, xanchor='center', x=0.5),
        bargap=0.05,
    )
    fig.update_xaxes(range=[x_lo, x_hi])
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    return fig


def plot_hypothesis_test_bar(p_value: float, question: str, alpha: float = 0.05) -> go.Figure:
    """
    Minitab-style horizontal p-value bar with Yes / No regions.
    The bar fills from left (p = 0) to the p-value position, coloured
    blue when p < alpha (Yes) and with a red boundary line at alpha.
    """
    max_x = max(0.5, p_value * 1.3, alpha * 5)
    bar_len = min(p_value, max_x)

    fig = go.Figure()

    # shaded significance region [0, alpha]
    fig.add_shape(
        type='rect', x0=0, x1=alpha, y0=-0.5, y1=0.5,
        fillcolor='rgba(70,130,180,0.15)', line=dict(width=0),
    )

    # p-value bar
    bar_colour = 'steelblue' if p_value < alpha else 'lightsteelblue'
    fig.add_trace(go.Bar(
        x=[bar_len], y=[0],
        orientation='h',
        marker_color=bar_colour,
        marker_line=dict(color='steelblue', width=1.5),
        width=0.6,
        showlegend=False,
    ))

    # alpha boundary line
    fig.add_shape(
        type='line', x0=alpha, x1=alpha, y0=-0.55, y1=0.55,
        line=dict(color='red', width=2),
    )

    # p-value annotation
    conclusion = "Yes" if p_value < alpha else "No"
    fig.add_annotation(
        x=bar_len, y=0,
        text=f"  P = {p_value:.3f}",
        showarrow=False, xanchor='left',
        font=dict(size=12, color='black', family='Arial'),
    )

    # axis tick labels
    tick_vals = [0]
    tick_vals += [round(alpha * k, 3) for k in [1, 2, 4, 6, 10] if round(alpha * k, 3) <= max_x]
    tick_vals = sorted(set(tick_vals))

    fig.update_layout(
        height=110,
        margin=dict(l=5, r=5, t=28, b=5),
        template='plotly_white',
        xaxis=dict(
            range=[0, max_x], tickvals=tick_vals,
            tickfont=dict(size=10),
            showgrid=False,
        ),
        yaxis=dict(
            showticklabels=False, showgrid=False,
            zeroline=False, range=[-0.7, 0.7],
        ),
        title=dict(text=question, font=dict(size=12), x=0.01, xanchor='left'),
        annotations=[
            dict(x=0, y=-0.65, text="<b>Yes</b>", showarrow=False,
                 xref='x', yref='y', font=dict(size=11, color='#2c5f8a'), xanchor='left'),
            dict(x=max_x, y=-0.65, text="<b>No</b>", showarrow=False,
                 xref='x', yref='y', font=dict(size=11, color='#8a2c2c'), xanchor='right'),
        ],
    )
    return fig


# =============================================================================
# BEFORE / AFTER  I-MR  CHART  –  Minitab-style
# =============================================================================

# SPC constants for moving-range span of 2
_D2 = 1.128      # d₂  – unbiasing constant for MR with n = 2
_D4 = 3.267      # D₄  – upper control limit factor for MR with n = 2


def _within_stage_stats(arr: np.ndarray) -> Dict:
    """
    Compute within-stage statistics for an I-MR chart.

    StDev(Within) = MR̄ / d₂   (d₂ = 1.128 for span 2)
    StDev(Overall) = sample SD (ddof = 1)
    """
    arr = np.asarray(arr, dtype=float)
    mr = np.abs(np.diff(arr))                 # moving ranges
    mr_bar = mr.mean() if len(mr) > 0 else 0.0
    sd_within = mr_bar / _D2
    sd_overall = float(np.std(arr, ddof=1))
    return {
        'n': len(arr),
        'mean': float(arr.mean()),
        'mr_bar': float(mr_bar),
        'sd_within': float(sd_within),
        'sd_overall': float(sd_overall),
        'values': arr,
        'mr': mr,
    }


def perform_imr_analysis(
    before: np.ndarray,
    after: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    Minitab-style Before / After I-MR analysis.

    * Control limits are computed from the **After** stage (Minitab default).
    * F-test (one-sided): was SD reduced?
    * Welch t-test (two-sided): did the mean change?
    * Directional comment: was the mean shift higher or lower?
    """
    before = np.asarray(before, dtype=float)
    after  = np.asarray(after,  dtype=float)
    before = before[~np.isnan(before)]
    after  = after[~np.isnan(after)]

    if len(before) < 2 or len(after) < 2:
        raise ValueError("Both stages need at least 2 observations.")

    b = _within_stage_stats(before)
    a = _within_stage_stats(after)

    # --- control limits (from After stage) -----------------------------------
    x_bar = a['mean']
    sd_w   = a['sd_within']
    mr_bar = a['mr_bar']

    i_ucl = x_bar + 3.0 * sd_w
    i_lcl = x_bar - 3.0 * sd_w
    mr_ucl = _D4 * mr_bar
    mr_lcl = 0.0

    # --- hypothesis tests ----------------------------------------------------
    # F-test: was SD reduced? (H₁: σ_before > σ_after, one-sided upper)
    f_stat = b['sd_overall']**2 / a['sd_overall']**2
    df1, df2 = b['n'] - 1, a['n'] - 1
    f_p = 1.0 - f_dist.cdf(f_stat, df1, df2)
    sd_reduced = f_p < alpha

    # Welch t-test: did mean change? (two-sided)
    t_res = stats.ttest_ind(before, after, equal_var=False)
    t_p = t_res.pvalue
    mean_changed = t_p < alpha

    # --- comments (Minitab wording) ------------------------------------------
    comments = [
        "After a process change, you may want to test whether the standard "
        "deviation or mean changed:"
    ]
    if sd_reduced:
        comments.append(
            f"The standard deviation was reduced significantly (p < {alpha})."
        )
    else:
        comments.append(
            f"The standard deviation was not reduced significantly (p > {alpha})."
        )

    if mean_changed:
        direction = "higher" if a['mean'] > b['mean'] else "lower"
        comments.append(
            f"The mean is significantly {direction} (p < {alpha}). "
            "Make sure the direction of the shift is an improvement."
        )
        comments.append(
            "Consider whether the change in the mean has practical implications."
        )
    else:
        comments.append(
            f"The process mean did not change significantly (p > {alpha})."
        )

    return {
        'before': b,
        'after':  a,
        'control_limits': {
            'i_center': x_bar,
            'i_ucl': i_ucl,
            'i_lcl': i_lcl,
            'mr_center': mr_bar,
            'mr_ucl': mr_ucl,
            'mr_lcl': mr_lcl,
        },
        'tests': {
            'f_test': {'statistic': f_stat, 'p_value': f_p, 'sd_reduced': sd_reduced},
            't_test': {'statistic': t_res.statistic, 'p_value': t_p, 'mean_changed': mean_changed},
        },
        'comments': comments,
    }


def plot_imr_chart(
    result: Dict,
    before_label: str = "Before",
    after_label: str = "After",
    response_label: str = "Individual Value",
) -> go.Figure:
    """
    Minitab-style Before/After I-MR chart.

    Top subplot  : Individuals chart with UCL / X̄ / LCL
    Bottom subplot: Moving-Range chart with UCL / MR̄ / LCL = 0
    Both have a vertical dashed separator between stages and stage labels.
    """
    before_vals = result['before']['values']
    after_vals  = result['after']['values']
    cl = result['control_limits']

    n_before = len(before_vals)
    n_after  = len(after_vals)
    n_total  = n_before + n_after

    # observation index (1-based, like Minitab)
    obs = np.arange(1, n_total + 1)
    all_vals = np.concatenate([before_vals, after_vals])

    # moving ranges (within each stage – no MR across boundary)
    mr_before = np.abs(np.diff(before_vals))
    mr_after  = np.abs(np.diff(after_vals))
    # indices for MR plot (MR_i corresponds to obs i, starting at i=2)
    mr_obs = np.concatenate([
        np.arange(2, n_before + 1),
        np.arange(n_before + 2, n_total + 1),
    ])
    mr_vals = np.concatenate([mr_before, mr_after])

    # boundary position (between last before and first after observation)
    boundary_x = n_before + 0.5

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.55, 0.45],
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    # ---- colours matching Minitab -------------------------------------------
    CL_GREEN  = '#2ca02c'    # UCL
    CL_BLUE   = '#1f77b4'    # centre line
    CL_RED    = '#d62728'    # LCL
    PT_BLACK  = '#222222'    # data points
    LINE_GREY = '#888888'    # connecting segments

    # ===================== TOP: Individuals chart ============================
    # data points + connecting lines
    fig.add_trace(go.Scatter(
        x=obs[:n_before], y=before_vals,
        mode='lines+markers',
        line=dict(color=LINE_GREY, width=1),
        marker=dict(color=PT_BLACK, size=4),
        name=before_label,
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=obs[n_before:], y=after_vals,
        mode='lines+markers',
        line=dict(color=LINE_GREY, width=1),
        marker=dict(color=PT_BLACK, size=4),
        name=after_label,
        showlegend=False,
    ), row=1, col=1)

    # control limit lines (full width)
    for y_val, colour, dash, label in [
        (cl['i_ucl'],    CL_GREEN, 'solid', f"UCL={cl['i_ucl']:.2f}"),
        (cl['i_center'], CL_BLUE,  'solid', f"X̄={cl['i_center']:.2f}"),
        (cl['i_lcl'],    CL_RED,   'solid', f"LCL={cl['i_lcl']:.2f}"),
    ]:
        fig.add_shape(type='line', x0=0.5, x1=n_total + 0.5,
                      y0=y_val, y1=y_val,
                      line=dict(color=colour, width=1.5, dash=dash),
                      row=1, col=1)
        # annotation on the right
        fig.add_annotation(
            x=n_total + 0.8, y=y_val, text=label,
            showarrow=False, xanchor='left',
            font=dict(size=10, color=colour),
            row=1, col=1,
        )

    # vertical stage boundary
    i_ymin = min(all_vals.min(), cl['i_lcl']) - 1
    i_ymax = max(all_vals.max(), cl['i_ucl']) + 1
    fig.add_shape(type='line', x0=boundary_x, x1=boundary_x,
                  y0=i_ymin, y1=i_ymax,
                  line=dict(color='steelblue', width=1.5, dash='dash'),
                  row=1, col=1)

    # stage labels
    fig.add_annotation(
        x=(1 + n_before) / 2, y=i_ymax,
        text=f"<b>{before_label}</b>", showarrow=False,
        font=dict(size=12, color='#333'), yanchor='bottom',
        row=1, col=1,
    )
    fig.add_annotation(
        x=n_before + (1 + n_after) / 2, y=i_ymax,
        text=f"<b>{after_label}</b>", showarrow=False,
        font=dict(size=12, color='#333'), yanchor='bottom',
        row=1, col=1,
    )

    fig.update_yaxes(title_text=response_label, row=1, col=1,
                     range=[i_ymin, i_ymax + 2])

    # ===================== BOTTOM: Moving Range chart ========================
    # Before MR segment
    if len(mr_before) > 0:
        fig.add_trace(go.Scatter(
            x=np.arange(2, n_before + 1), y=mr_before,
            mode='lines+markers',
            line=dict(color=LINE_GREY, width=1),
            marker=dict(color=PT_BLACK, size=4),
            showlegend=False,
        ), row=2, col=1)

    # After MR segment
    if len(mr_after) > 0:
        fig.add_trace(go.Scatter(
            x=np.arange(n_before + 2, n_total + 1), y=mr_after,
            mode='lines+markers',
            line=dict(color=LINE_GREY, width=1),
            marker=dict(color=PT_BLACK, size=4),
            showlegend=False,
        ), row=2, col=1)

    # MR control limit lines
    mr_ymax = max(mr_vals.max() if len(mr_vals) else 1, cl['mr_ucl']) * 1.15
    for y_val, colour, label in [
        (cl['mr_ucl'],    CL_GREEN, f"UCL={cl['mr_ucl']:.3f}"),
        (cl['mr_center'], CL_BLUE,  f"MR̄={cl['mr_center']:.3f}"),
        (cl['mr_lcl'],    CL_RED,   "LCL=0"),
    ]:
        fig.add_shape(type='line', x0=0.5, x1=n_total + 0.5,
                      y0=y_val, y1=y_val,
                      line=dict(color=colour, width=1.5),
                      row=2, col=1)
        fig.add_annotation(
            x=n_total + 0.8, y=y_val, text=label,
            showarrow=False, xanchor='left',
            font=dict(size=10, color=colour),
            row=2, col=1,
        )

    # MR boundary
    fig.add_shape(type='line', x0=boundary_x, x1=boundary_x,
                  y0=0, y1=mr_ymax,
                  line=dict(color='steelblue', width=1.5, dash='dash'),
                  row=2, col=1)

    fig.update_yaxes(title_text="Moving Range", row=2, col=1,
                     range=[-0.1, mr_ymax])

    # ---- layout -------------------------------------------------------------
    fig.update_xaxes(
        title_text="Observation", row=2, col=1,
        range=[0, n_total + max(6, int(n_total * 0.06))],   # extra space for annotations
    )
    fig.update_xaxes(range=[0, n_total + max(6, int(n_total * 0.06))], row=1, col=1)

    fig.update_layout(
        height=560,
        template='plotly_white',
        margin=dict(l=60, r=120, t=20, b=40),
        showlegend=False,
    )
    return fig
