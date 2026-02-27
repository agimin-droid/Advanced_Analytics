"""
EDA Calculations Module
=======================

Statistical calculations for the Minitab-style Summary Report.

Provides:
- Anderson-Darling normality test
- 95% Confidence intervals for Mean, Median, and StDev
- Full descriptive statistics (mean, median, quartiles, skewness, kurtosis, etc.)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Tuple, Union


# ──────────────────────────────────────────────
#  NORMALITY TEST
# ──────────────────────────────────────────────

def anderson_darling_test(data: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
    """
    Anderson-Darling normality test.

    Uses scipy.stats.anderson and maps the critical value table to
    approximate p-value brackets, matching Minitab conventions.

    Parameters
    ----------
    data : array-like
        Numeric data (NaN values are dropped automatically)

    Returns
    -------
    dict
        Keys: 'statistic' (A-Squared), 'p_value' (float or '<0.005'),
              'p_label' (display string), 'reject_h0' (bool)

    Notes
    -----
    scipy returns critical values at [15%, 10%, 5%, 2.5%, 1%].
    P-value is approximated from those brackets.
    """
    data_clean = _clean(data)

    # Use interpolate method if available (SciPy >= 1.17), else legacy API
    try:
        result = stats.anderson(data_clean, dist='norm', method='interpolate')
        a_sq = result.statistic
        p_value_raw = float(result.pvalue)
        # Round to a clean display label
        if p_value_raw < 0.005:
            p_label, p_value = "<0.005", 0.005
        elif p_value_raw < 0.01:
            p_label, p_value = f"{p_value_raw:.4f}", p_value_raw
        else:
            p_label, p_value = f"{p_value_raw:.4f}", p_value_raw
    except TypeError:
        # SciPy < 1.17 fallback: map critical-value table to p-value brackets
        result = stats.anderson(data_clean, dist='norm')
        a_sq = result.statistic
        sig_levels = [0.15, 0.10, 0.05, 0.025, 0.01]
        crits = result.critical_values
        p_label = "<0.005"
        p_value = 0.005
        for level, crit in zip(sig_levels, crits):
            if a_sq < crit:
                p_label = f">{level:.3f}"
                p_value = level
                break

    # Minitab uses α=0.05 as rejection threshold
    reject_h0 = (p_value <= 0.05) if isinstance(p_value, float) else True

    return {
        'statistic': round(a_sq, 4),
        'p_value': p_value,
        'p_label': p_label,
        'reject_h0': reject_h0
    }


# ──────────────────────────────────────────────
#  CONFIDENCE INTERVALS
# ──────────────────────────────────────────────

def ci_for_mean(
    data: Union[np.ndarray, pd.Series],
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Confidence interval for the mean using t-distribution.

    Parameters
    ----------
    data : array-like
    confidence : float, default 0.95

    Returns
    -------
    (lower, upper) : tuple of float
    """
    data_clean = _clean(data)
    n = len(data_clean)
    mean = np.mean(data_clean)
    se = stats.sem(data_clean)
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    margin = t_crit * se
    return (round(mean - margin, 4), round(mean + margin, 4))


def ci_for_median(
    data: Union[np.ndarray, pd.Series],
    confidence: float = 0.95,
    n_bootstrap: int = 5000,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for the median.

    Parameters
    ----------
    data : array-like
    confidence : float, default 0.95
    n_bootstrap : int, default 5000
    random_state : int, default 42

    Returns
    -------
    (lower, upper) : tuple of float
    """
    data_clean = _clean(data)
    rng = np.random.default_rng(random_state)
    boot_medians = np.array([
        np.median(rng.choice(data_clean, size=len(data_clean), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1 - confidence
    lower = float(np.percentile(boot_medians, 100 * alpha / 2))
    upper = float(np.percentile(boot_medians, 100 * (1 - alpha / 2)))
    return (round(lower, 4), round(upper, 4))


def ci_for_stdev(
    data: Union[np.ndarray, pd.Series],
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Confidence interval for the standard deviation using chi-squared distribution.

    Parameters
    ----------
    data : array-like
    confidence : float, default 0.95

    Returns
    -------
    (lower, upper) : tuple of float
    """
    data_clean = _clean(data)
    n = len(data_clean)
    s = np.std(data_clean, ddof=1)
    alpha = 1 - confidence
    chi2_lower = stats.chi2.ppf(alpha / 2, df=n - 1)
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, df=n - 1)
    lower = np.sqrt((n - 1) * s**2 / chi2_upper)
    upper = np.sqrt((n - 1) * s**2 / chi2_lower)
    return (round(lower, 4), round(upper, 4))


# ──────────────────────────────────────────────
#  DESCRIPTIVE STATISTICS
# ──────────────────────────────────────────────

def descriptive_statistics(data: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
    """
    Comprehensive descriptive statistics matching the Minitab Summary Report.

    Parameters
    ----------
    data : array-like

    Returns
    -------
    dict with keys:
        n, n_missing, mean, stdev, variance, skewness, kurtosis,
        minimum, q1, median, q3, maximum,
        ci_mean, ci_median, ci_stdev,
        ad_statistic, ad_p_label, ad_reject
    """
    arr = np.asarray(data).flatten()
    n_total = len(arr)
    n_missing = int(np.sum(np.isnan(arr)))
    data_clean = arr[~np.isnan(arr)]
    n = len(data_clean)

    mean_val   = float(np.mean(data_clean))
    stdev_val  = float(np.std(data_clean, ddof=1))
    var_val    = float(np.var(data_clean, ddof=1))
    skew_val   = float(stats.skew(data_clean))
    kurt_val   = float(stats.kurtosis(data_clean))       # excess kurtosis

    minimum  = float(np.min(data_clean))
    q1       = float(np.percentile(data_clean, 25))
    median   = float(np.median(data_clean))
    q3       = float(np.percentile(data_clean, 75))
    maximum  = float(np.max(data_clean))

    ci_mean   = ci_for_mean(data_clean)
    ci_median = ci_for_median(data_clean)
    ci_stdev  = ci_for_stdev(data_clean)

    ad = anderson_darling_test(data_clean)

    return {
        'n':          n,
        'n_missing':  n_missing,
        'mean':       round(mean_val,  4),
        'stdev':      round(stdev_val, 4),
        'variance':   round(var_val,   4),
        'skewness':   round(skew_val,  6),
        'kurtosis':   round(kurt_val,  6),
        'minimum':    round(minimum,   4),
        'q1':         round(q1,        4),
        'median':     round(median,    4),
        'q3':         round(q3,        4),
        'maximum':    round(maximum,   4),
        'ci_mean':    ci_mean,
        'ci_median':  ci_median,
        'ci_stdev':   ci_stdev,
        'ad_statistic': ad['statistic'],
        'ad_p_label':   ad['p_label'],
        'ad_reject':    ad['reject_h0'],
    }


def run_eda_for_all_columns(
    dataframe: pd.DataFrame,
    numeric_only: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Run full EDA calculations for every column in a DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
    numeric_only : bool, default True
        Skip non-numeric columns when True.

    Returns
    -------
    dict  {column_name: descriptive_statistics_dict}
    """
    results = {}
    cols = dataframe.select_dtypes(include=[np.number]).columns if numeric_only \
           else dataframe.columns

    for col in cols:
        try:
            results[col] = descriptive_statistics(dataframe[col])
        except Exception as e:
            results[col] = {'error': str(e)}

    return results


# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────

def _clean(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """Return a 1-D float array with NaN values removed."""
    arr = np.asarray(data).flatten().astype(float)
    return arr[~np.isnan(arr)]
