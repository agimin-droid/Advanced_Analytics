"""
Two-Sample t-Test – Full Minitab-Style Implementation + Sample Size Calculator
"""

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from typing import Dict, Optional


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
    """Schuirmann’s TOST sample size calculation."""
    from scipy.stats import norm
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
    pooled_var = ((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2)
    return np.sqrt(pooled_var)