"""
Two-Sample T-Test Calculations Module

Replicates Minitab's 2-Sample t-test functionality:
- Pooled (equal variances) and Welch (unequal variances) t-tests
- One-sided and two-sided hypothesis tests
- Confidence intervals for difference of means
- Levene's test for equality of variances
- Normality tests (Shapiro-Wilk, Anderson-Darling)
- Effect size measures (Cohen's d, Hedges' g)

Reference: Minitab Statistical Software, Two-Sample t-Test
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional, Tuple, Union


# ========== CORE T-TEST ==========

def two_sample_ttest(
    sample1: Union[np.ndarray, pd.Series],
    sample2: Union[np.ndarray, pd.Series],
    alternative: str = "two-sided",
    equal_var: bool = False,
    confidence_level: float = 0.95,
    hypothesized_difference: float = 0.0
) -> Dict[str, Any]:
    """
    Perform two-sample t-test replicating Minitab output.

    Parameters
    ----------
    sample1 : array-like
        First sample data
    sample2 : array-like
        Second sample data
    alternative : str
        'two-sided', 'less', or 'greater'
        - 'less': H1: μ1 - μ2 < hypothesized_difference
        - 'greater': H1: μ1 - μ2 > hypothesized_difference
    equal_var : bool
        If True, use pooled variance (assumes equal variances).
        If False, use Welch's t-test (default, Minitab default).
    confidence_level : float
        Confidence level for CI (default 0.95 = 95%)
    hypothesized_difference : float
        Hypothesized difference μ1 - μ2 (default 0)

    Returns
    -------
    dict
        Complete test results matching Minitab output structure:
        - descriptive: per-sample statistics
        - test_results: t-statistic, df, p-value
        - ci: confidence interval for difference
        - effect_size: Cohen's d and Hedges' g
        - method: test method description
    """
    # Clean data
    s1 = np.asarray(sample1, dtype=float).flatten()
    s2 = np.asarray(sample2, dtype=float).flatten()
    s1 = s1[~np.isnan(s1)]
    s2 = s2[~np.isnan(s2)]

    n1, n2 = len(s1), len(s2)

    if n1 < 2 or n2 < 2:
        raise ValueError(
            f"Each sample must have at least 2 observations. "
            f"Got n1={n1}, n2={n2}."
        )

    # Descriptive statistics
    mean1, mean2 = np.mean(s1), np.mean(s2)
    std1, std2 = np.std(s1, ddof=1), np.std(s2, ddof=1)
    se1, se2 = std1 / np.sqrt(n1), std2 / np.sqrt(n2)

    # Observed difference
    diff = mean1 - mean2

    # ===== T-STATISTIC AND DEGREES OF FREEDOM =====
    if equal_var:
        # Pooled t-test
        sp2 = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
        sp = np.sqrt(sp2)
        se_diff = sp * np.sqrt(1/n1 + 1/n2)
        df = n1 + n2 - 2
        method = "Two-Sample T-Test (Pooled, Equal Variances Assumed)"
    else:
        # Welch's t-test (Minitab default)
        se_diff = np.sqrt(std1**2/n1 + std2**2/n2)
        # Welch-Satterthwaite degrees of freedom
        numerator = (std1**2/n1 + std2**2/n2)**2
        denominator = (std1**2/n1)**2/(n1 - 1) + (std2**2/n2)**2/(n2 - 1)
        df = numerator / denominator
        method = "Two-Sample T-Test (Welch's, Equal Variances Not Assumed)"

    # T-statistic
    t_stat = (diff - hypothesized_difference) / se_diff

    # ===== P-VALUE =====
    if alternative == "two-sided":
        p_value = 2 * stats.t.sf(np.abs(t_stat), df)
        alt_description = f"Difference ≠ {hypothesized_difference}"
    elif alternative == "less":
        p_value = stats.t.cdf(t_stat, df)
        alt_description = f"Difference < {hypothesized_difference}"
    elif alternative == "greater":
        p_value = stats.t.sf(t_stat, df)
        alt_description = f"Difference > {hypothesized_difference}"
    else:
        raise ValueError(f"alternative must be 'two-sided', 'less', or 'greater'. Got '{alternative}'.")

    # ===== CONFIDENCE INTERVAL =====
    alpha = 1 - confidence_level
    if alternative == "two-sided":
        t_crit = stats.t.ppf(1 - alpha/2, df)
        ci_lower = diff - t_crit * se_diff
        ci_upper = diff + t_crit * se_diff
    elif alternative == "less":
        t_crit = stats.t.ppf(1 - alpha, df)
        ci_lower = -np.inf
        ci_upper = diff + t_crit * se_diff
    elif alternative == "greater":
        t_crit = stats.t.ppf(1 - alpha, df)
        ci_lower = diff - t_crit * se_diff
        ci_upper = np.inf

    # ===== EFFECT SIZE =====
    # Cohen's d (uses pooled std regardless of test type)
    pooled_std = np.sqrt(((n1 - 1)*std1**2 + (n2 - 1)*std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else np.nan

    # Hedges' g (bias-corrected Cohen's d)
    correction = 1 - 3 / (4*(n1 + n2 - 2) - 1)
    hedges_g = cohens_d * correction

    return {
        'descriptive': {
            'sample1': {
                'n': n1,
                'mean': mean1,
                'std_dev': std1,
                'se_mean': se1
            },
            'sample2': {
                'n': n2,
                'mean': mean2,
                'std_dev': std2,
                'se_mean': se2
            }
        },
        'test_results': {
            'difference': diff,
            'se_difference': se_diff,
            't_statistic': t_stat,
            'df': df,
            'p_value': p_value,
            'hypothesized_difference': hypothesized_difference,
            'alternative': alternative,
            'alt_description': alt_description
        },
        'ci': {
            'confidence_level': confidence_level,
            'lower': ci_lower,
            'upper': ci_upper
        },
        'effect_size': {
            'cohens_d': cohens_d,
            'hedges_g': hedges_g,
            'pooled_std': pooled_std
        },
        'method': method,
        'equal_var': equal_var
    }


# ========== VARIANCE EQUALITY TESTS ==========

def test_equal_variances(
    sample1: Union[np.ndarray, pd.Series],
    sample2: Union[np.ndarray, pd.Series]
) -> Dict[str, Any]:
    """
    Test equality of variances between two samples.

    Replicates Minitab's variance tests:
    - F-test (ratio of variances)
    - Levene's test (robust to non-normality)
    - Bartlett's test (sensitive to normality)

    Parameters
    ----------
    sample1, sample2 : array-like
        Sample data

    Returns
    -------
    dict
        Results for each test: statistic, p-value, conclusion
    """
    s1 = np.asarray(sample1, dtype=float).flatten()
    s2 = np.asarray(sample2, dtype=float).flatten()
    s1 = s1[~np.isnan(s1)]
    s2 = s2[~np.isnan(s2)]

    results = {}

    # F-test for equality of variances
    var1, var2 = np.var(s1, ddof=1), np.var(s2, ddof=1)
    f_stat = var1 / var2 if var2 > 0 else np.inf
    df1, df2 = len(s1) - 1, len(s2) - 1
    # Two-tailed F-test
    p_f = 2 * min(stats.f.cdf(f_stat, df1, df2), stats.f.sf(f_stat, df1, df2))

    results['f_test'] = {
        'test_name': "F-Test",
        'statistic': f_stat,
        'df1': df1,
        'df2': df2,
        'p_value': p_f,
        'var1': var1,
        'var2': var2,
        'ratio': f_stat
    }

    # Levene's test (robust)
    lev_stat, lev_p = stats.levene(s1, s2, center='median')
    results['levene'] = {
        'test_name': "Levene's Test",
        'statistic': lev_stat,
        'p_value': lev_p
    }

    # Bartlett's test (assumes normality)
    try:
        bart_stat, bart_p = stats.bartlett(s1, s2)
        results['bartlett'] = {
            'test_name': "Bartlett's Test",
            'statistic': bart_stat,
            'p_value': bart_p
        }
    except Exception:
        results['bartlett'] = {
            'test_name': "Bartlett's Test",
            'statistic': np.nan,
            'p_value': np.nan
        }

    return results


# ========== NORMALITY TESTS ==========

def test_normality(
    sample: Union[np.ndarray, pd.Series],
    sample_name: str = "Sample"
) -> Dict[str, Any]:
    """
    Test normality of a sample.

    Replicates Minitab's normality tests:
    - Shapiro-Wilk (best for small samples)
    - Anderson-Darling (general purpose)
    - D'Agostino-Pearson (combines skewness + kurtosis)

    Parameters
    ----------
    sample : array-like
        Sample data
    sample_name : str
        Label for the sample

    Returns
    -------
    dict
        Results for each normality test
    """
    s = np.asarray(sample, dtype=float).flatten()
    s = s[~np.isnan(s)]

    results = {'sample_name': sample_name, 'n': len(s)}

    # Shapiro-Wilk
    if 3 <= len(s) <= 5000:
        sw_stat, sw_p = stats.shapiro(s)
        results['shapiro_wilk'] = {
            'test_name': "Shapiro-Wilk",
            'statistic': sw_stat,
            'p_value': sw_p
        }
    else:
        results['shapiro_wilk'] = {
            'test_name': "Shapiro-Wilk",
            'statistic': np.nan,
            'p_value': np.nan,
            'note': f"Requires 3-5000 observations (n={len(s)})"
        }

    # Anderson-Darling
    try:
        ad_result = stats.anderson(s, dist='norm')
        # Convert AD statistic to approximate p-value
        ad_stat = ad_result.statistic
        # Use critical values and significance levels
        # Anderson-Darling critical values at 15%, 10%, 5%, 2.5%, 1%
        sig_levels = ad_result.significance_level / 100  # Convert to proportions
        crit_values = ad_result.critical_values

        # Determine approximate p-value
        if ad_stat < crit_values[0]:
            ad_p = sig_levels[0]  # p > 0.15
        elif ad_stat > crit_values[-1]:
            ad_p = sig_levels[-1]  # p < 0.01
        else:
            # Interpolate
            for i in range(len(crit_values) - 1):
                if crit_values[i] <= ad_stat <= crit_values[i+1]:
                    ad_p = sig_levels[i+1]
                    break
            else:
                ad_p = np.nan

        results['anderson_darling'] = {
            'test_name': "Anderson-Darling",
            'statistic': ad_stat,
            'p_value': ad_p,
            'critical_values': dict(zip(
                [f"{sl}%" for sl in ad_result.significance_level],
                ad_result.critical_values
            ))
        }
    except Exception:
        results['anderson_darling'] = {
            'test_name': "Anderson-Darling",
            'statistic': np.nan,
            'p_value': np.nan
        }

    # D'Agostino-Pearson (n >= 8)
    if len(s) >= 8:
        try:
            dp_stat, dp_p = stats.normaltest(s)
            results['dagostino_pearson'] = {
                'test_name': "D'Agostino-Pearson",
                'statistic': dp_stat,
                'p_value': dp_p
            }
        except Exception:
            results['dagostino_pearson'] = {
                'test_name': "D'Agostino-Pearson",
                'statistic': np.nan,
                'p_value': np.nan
            }
    else:
        results['dagostino_pearson'] = {
            'test_name': "D'Agostino-Pearson",
            'statistic': np.nan,
            'p_value': np.nan,
            'note': f"Requires n >= 8 (n={len(s)})"
        }

    # Skewness and Kurtosis (Minitab-style descriptive)
    results['skewness'] = stats.skew(s)
    results['kurtosis'] = stats.kurtosis(s)

    return results


# ========== POWER ANALYSIS ==========

def ttest_power(
    n1: int,
    n2: int,
    effect_size: float,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> float:
    """
    Approximate power for two-sample t-test.

    Uses non-central t-distribution.

    Parameters
    ----------
    n1, n2 : int
        Sample sizes
    effect_size : float
        Cohen's d (standardized difference)
    alpha : float
        Significance level
    alternative : str
        'two-sided', 'less', or 'greater'

    Returns
    -------
    float
        Statistical power (0-1)
    """
    df = n1 + n2 - 2
    # Non-centrality parameter
    ncp = effect_size * np.sqrt(n1 * n2 / (n1 + n2))

    if alternative == "two-sided":
        t_crit = stats.t.ppf(1 - alpha/2, df)
        power = (
            stats.nct.sf(t_crit, df, ncp) +
            stats.nct.cdf(-t_crit, df, ncp)
        )
    elif alternative == "greater":
        t_crit = stats.t.ppf(1 - alpha, df)
        power = stats.nct.sf(t_crit, df, ncp)
    elif alternative == "less":
        t_crit = stats.t.ppf(alpha, df)
        power = stats.nct.cdf(t_crit, df, ncp)
    else:
        power = np.nan

    return power


# ========== MINITAB-STYLE OUTPUT FORMATTER ==========

def format_minitab_output(results: Dict[str, Any], label1: str = "Sample 1", label2: str = "Sample 2") -> str:
    """
    Format t-test results as Minitab-style text output.

    Parameters
    ----------
    results : dict
        Output from two_sample_ttest()
    label1, label2 : str
        Labels for the two samples

    Returns
    -------
    str
        Formatted text output matching Minitab style
    """
    desc = results['descriptive']
    test = results['test_results']
    ci = results['ci']
    eff = results['effect_size']

    lines = []
    lines.append("=" * 65)
    lines.append(results['method'])
    lines.append("=" * 65)
    lines.append("")

    # Descriptive Statistics
    lines.append("Descriptive Statistics")
    lines.append("")
    lines.append(f"{'Sample':<20} {'N':>6} {'Mean':>12} {'StDev':>12} {'SE Mean':>12}")
    lines.append("-" * 65)
    lines.append(
        f"{label1:<20} {desc['sample1']['n']:>6} "
        f"{desc['sample1']['mean']:>12.4f} {desc['sample1']['std_dev']:>12.4f} "
        f"{desc['sample1']['se_mean']:>12.4f}"
    )
    lines.append(
        f"{label2:<20} {desc['sample2']['n']:>6} "
        f"{desc['sample2']['mean']:>12.4f} {desc['sample2']['std_dev']:>12.4f} "
        f"{desc['sample2']['se_mean']:>12.4f}"
    )
    lines.append("")

    # Estimation for Difference
    lines.append("Estimation for Difference")
    lines.append("")
    lines.append(
        f"  Difference: {test['difference']:.4f}"
    )
    lines.append(
        f"  SE of Difference: {test['se_difference']:.4f}"
    )

    ci_pct = int(ci['confidence_level'] * 100)
    if np.isfinite(ci['lower']) and np.isfinite(ci['upper']):
        lines.append(
            f"  {ci_pct}% CI for Difference: ({ci['lower']:.4f}, {ci['upper']:.4f})"
        )
    elif not np.isfinite(ci['lower']):
        lines.append(
            f"  {ci_pct}% Upper Bound for Difference: {ci['upper']:.4f}"
        )
    else:
        lines.append(
            f"  {ci_pct}% Lower Bound for Difference: {ci['lower']:.4f}"
        )
    lines.append("")

    # Test
    null_str = f"Difference = {test['hypothesized_difference']}"
    lines.append("Test")
    lines.append("")
    lines.append(f"  Null hypothesis:        H₀: {null_str}")
    lines.append(f"  Alternative hypothesis: H₁: {test['alt_description']}")
    lines.append("")
    lines.append(f"  T-Value: {test['t_statistic']:.4f}")
    lines.append(f"  DF:      {test['df']:.2f}")
    lines.append(f"  P-Value: {test['p_value']:.6f}")
    lines.append("")

    # Effect Size
    lines.append("Effect Size")
    lines.append("")
    lines.append(f"  Cohen's d: {eff['cohens_d']:.4f}")
    lines.append(f"  Hedges' g: {eff['hedges_g']:.4f}")

    # Interpret Cohen's d
    d_abs = abs(eff['cohens_d'])
    if d_abs < 0.2:
        interpretation = "negligible"
    elif d_abs < 0.5:
        interpretation = "small"
    elif d_abs < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    lines.append(f"  Interpretation: {interpretation} effect")
    lines.append("")
    lines.append("=" * 65)

    return "\n".join(lines)


# ========== BATCH COMPARISON ==========

def pairwise_ttest(
    dataframe: pd.DataFrame,
    value_column: str,
    group_column: str,
    equal_var: bool = False,
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Perform pairwise two-sample t-tests across all group pairs.

    Useful for comparing multiple batches / categories.

    Parameters
    ----------
    dataframe : pd.DataFrame
    value_column : str
        Numeric column with values
    group_column : str
        Categorical column defining groups
    equal_var : bool
        Assume equal variances
    confidence_level : float

    Returns
    -------
    pd.DataFrame
        Pairwise comparison table with t-stat, df, p-value per pair
    """
    groups = sorted(dataframe[group_column].dropna().unique())
    results_list = []

    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            g1 = dataframe[dataframe[group_column] == groups[i]][value_column].dropna().values
            g2 = dataframe[dataframe[group_column] == groups[j]][value_column].dropna().values

            if len(g1) < 2 or len(g2) < 2:
                continue

            res = two_sample_ttest(
                g1, g2,
                alternative="two-sided",
                equal_var=equal_var,
                confidence_level=confidence_level
            )

            results_list.append({
                'Group 1': groups[i],
                'Group 2': groups[j],
                'N1': res['descriptive']['sample1']['n'],
                'N2': res['descriptive']['sample2']['n'],
                'Mean 1': res['descriptive']['sample1']['mean'],
                'Mean 2': res['descriptive']['sample2']['mean'],
                'Difference': res['test_results']['difference'],
                'T-Statistic': res['test_results']['t_statistic'],
                'DF': res['test_results']['df'],
                'P-Value': res['test_results']['p_value'],
                'CI Lower': res['ci']['lower'],
                'CI Upper': res['ci']['upper'],
                "Cohen's d": res['effect_size']['cohens_d']
            })

    return pd.DataFrame(results_list)
