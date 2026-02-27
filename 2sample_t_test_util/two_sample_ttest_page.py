"""
Two-Sample T-Test Page - Streamlit Interface

Replicates Minitab's 2-Sample t-Test capability:
- Tab 1: Test Setup & Results (core t-test)
- Tab 2: Graphs (Individual Value, Boxplot, Histogram, Q-Q)
- Tab 3: Assumptions (Normality tests, Variance equality)
- Tab 4: Report & Export (Minitab-style text report, CSV/Excel)

Data input modes (Minitab-compatible):
  A) Both samples in one column with a grouping column
  B) Each sample in a separate column
  C) Summarized data (n, mean, StDev per group)

Author: ChemometricSolutions
Version: 1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from io import BytesIO

# Import t-test modules (from univariate_utils subfolder)
try:
    from univariate_utils.ttest_calculations import (
        two_sample_ttest,
        test_equal_variances,
        test_normality,
        ttest_power,
        format_minitab_output,
        pairwise_ttest
    )
    TTEST_CALC_AVAILABLE = True
except ImportError as e:
    TTEST_CALC_AVAILABLE = False
    import sys
    print(f"⚠️ ttest_calculations import failed: {e}", file=sys.stderr)

try:
    from univariate_utils.ttest_plots import (
        plot_individual_value,
        plot_comparative_boxplot,
        plot_histogram_overlay,
        plot_qq_by_group,
        plot_interval_plot,
        plot_ttest_fourplot,
        plot_test_report
    )
    TTEST_PLOTS_AVAILABLE = True
except ImportError as e:
    TTEST_PLOTS_AVAILABLE = False
    import sys
    print(f"⚠️ ttest_plots import failed: {e}", file=sys.stderr)

# Import workspace utilities
try:
    from workspace_utils import get_workspace_datasets
    WORKSPACE_AVAILABLE = True
except ImportError:
    WORKSPACE_AVAILABLE = False


# ========== HELPERS ==========

def _get_numeric_columns(df: pd.DataFrame) -> list:
    """Return columns that can be converted to numeric."""
    numeric_cols = []
    for col in df.columns:
        try:
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().sum() >= 2:
                numeric_cols.append(col)
        except Exception:
            pass
    return numeric_cols


def _get_categorical_columns(df: pd.DataFrame, max_unique: int = 50) -> list:
    """Return columns suitable as grouping variables."""
    cat_cols = []
    for col in df.columns:
        n_unique = df[col].nunique()
        if 2 <= n_unique <= max_unique:
            cat_cols.append(col)
    return cat_cols


def _significance_badge(p_value: float, alpha: float) -> str:
    """Return colored significance badge."""
    if p_value < 0.001:
        return "🟢 **Highly Significant** (p < 0.001)"
    elif p_value < alpha:
        return f"🟢 **Significant** (p = {p_value:.4f} < α = {alpha})"
    elif p_value < alpha * 2:
        return f"🟡 **Marginally Non-Significant** (p = {p_value:.4f})"
    else:
        return f"🔴 **Not Significant** (p = {p_value:.4f} > α = {alpha})"


# ========== MAIN PAGE ==========

def show():
    """Main function - Two-Sample T-Test Page"""

    if not TTEST_CALC_AVAILABLE:
        st.error("❌ T-test calculation module not available. Check univariate_utils/ttest_calculations.py")
        return

    st.markdown("""
    # 🔬 Two-Sample T-Test

    **Compare means of two independent groups — Minitab-equivalent analysis**

    Features:
    - 📊 Pooled or Welch's t-test (equal/unequal variances)
    - 📐 One-sided and two-sided hypothesis tests
    - 📈 Confidence interval for difference of means
    - ✅ Normality and equal variance assumption checks
    - 📋 Minitab-style output report
    """)

    # ===== DATASET SELECTION =====
    st.markdown("---")
    st.subheader("1️⃣ Dataset Selection")

    if WORKSPACE_AVAILABLE:
        available_datasets = get_workspace_datasets()
        if not available_datasets:
            st.warning("⚠️ No datasets available. Please upload data in Data Handling first.")
            return

        dataset_name = st.selectbox(
            "Select dataset:",
            options=list(available_datasets.keys()),
            help="Choose from datasets in workspace",
            key="ttest_dataset"
        )
        selected_data = available_datasets[dataset_name]
    else:
        if 'current_data' not in st.session_state or st.session_state.current_data is None:
            st.error("❌ No data available. Load a dataset first.")
            return
        dataset_name = "Current Data"
        selected_data = st.session_state.current_data

    st.success(f"✅ **{dataset_name}** — {len(selected_data)} samples × {len(selected_data.columns)} variables")

    numeric_cols = _get_numeric_columns(selected_data)
    cat_cols = _get_categorical_columns(selected_data)

    if len(numeric_cols) == 0:
        st.error("❌ No numeric columns found in dataset.")
        return

    # ===== DATA INPUT MODE (MINITAB-STYLE) =====
    st.markdown("---")
    st.subheader("2️⃣ Data Configuration")

    input_mode = st.radio(
        "How are your samples arranged?",
        [
            "Both samples in one column (with grouping column)",
            "Each sample in its own column",
            "Summarized data (enter statistics directly)"
        ],
        key="ttest_input_mode",
        help=(
            "**One column + grouping**: Like Minitab's 'Samples in one column' — "
            "one column has all values, another column identifies the group.\n\n"
            "**Separate columns**: Each sample is a different column in your dataset.\n\n"
            "**Summarized data**: Enter n, mean, and StDev directly if you don't have raw data."
        )
    )

    sample1_data = None
    sample2_data = None
    label1 = "Sample 1"
    label2 = "Sample 2"

    # ---- MODE A: One column + grouping ----
    if input_mode == "Both samples in one column (with grouping column)":
        col_a, col_b = st.columns(2)

        with col_a:
            value_col = st.selectbox(
                "📈 Response (numeric) column:",
                numeric_cols,
                key="ttest_value_col"
            )

        with col_b:
            if not cat_cols:
                st.warning("⚠️ No suitable grouping columns found (need 2-50 unique values).")
                # Allow choosing any column
                group_col = st.selectbox(
                    "🏷️ Grouping column:",
                    selected_data.columns.tolist(),
                    key="ttest_group_col"
                )
            else:
                group_col = st.selectbox(
                    "🏷️ Grouping column:",
                    cat_cols,
                    key="ttest_group_col"
                )

        # Get unique groups
        groups = sorted(selected_data[group_col].dropna().unique())

        if len(groups) < 2:
            st.error(f"❌ Need at least 2 groups. Column '{group_col}' has {len(groups)} unique value(s).")
            return

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            group1 = st.selectbox("Group 1:", groups, index=0, key="ttest_g1")
        with col_g2:
            remaining = [g for g in groups if g != group1]
            group2 = st.selectbox("Group 2:", remaining, index=0, key="ttest_g2")

        label1 = str(group1)
        label2 = str(group2)

        # Extract data
        s1_raw = selected_data[selected_data[group_col] == group1][value_col]
        s2_raw = selected_data[selected_data[group_col] == group2][value_col]
        sample1_data = pd.to_numeric(s1_raw, errors='coerce').dropna().values
        sample2_data = pd.to_numeric(s2_raw, errors='coerce').dropna().values

    # ---- MODE B: Separate columns ----
    elif input_mode == "Each sample in its own column":
        col_a, col_b = st.columns(2)
        with col_a:
            col1 = st.selectbox("📈 Sample 1 column:", numeric_cols, index=0, key="ttest_col1")
        with col_b:
            default_idx = min(1, len(numeric_cols) - 1)
            col2 = st.selectbox("📈 Sample 2 column:", numeric_cols, index=default_idx, key="ttest_col2")

        if col1 == col2:
            st.warning("⚠️ Please select two different columns.")
            return

        label1 = str(col1)
        label2 = str(col2)

        sample1_data = pd.to_numeric(selected_data[col1], errors='coerce').dropna().values
        sample2_data = pd.to_numeric(selected_data[col2], errors='coerce').dropna().values

    # ---- MODE C: Summarized data ----
    else:
        st.info("Enter summary statistics for each sample. The test will be computed from these values.")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Sample 1**")
            n1_input = st.number_input("Sample size (n₁):", min_value=2, value=30, key="ttest_n1")
            mean1_input = st.number_input("Mean (x̄₁):", value=0.0, format="%.6f", key="ttest_mean1")
            std1_input = st.number_input("StDev (s₁):", min_value=0.0001, value=1.0, format="%.6f", key="ttest_std1")
            label1 = st.text_input("Label:", value="Sample 1", key="ttest_label1")

        with col_b:
            st.markdown("**Sample 2**")
            n2_input = st.number_input("Sample size (n₂):", min_value=2, value=30, key="ttest_n2")
            mean2_input = st.number_input("Mean (x̄₂):", value=0.0, format="%.6f", key="ttest_mean2")
            std2_input = st.number_input("StDev (s₂):", min_value=0.0001, value=1.0, format="%.6f", key="ttest_std2")
            label2 = st.text_input("Label:", value="Sample 2", key="ttest_label2")

        # Generate synthetic data matching the summary stats (for plots)
        # Use inverse normal CDF to create data with exact mean and std
        np.random.seed(42)
        sample1_data = np.random.normal(mean1_input, std1_input, int(n1_input))
        sample2_data = np.random.normal(mean2_input, std2_input, int(n2_input))
        # Adjust to exact mean and std
        sample1_data = (sample1_data - sample1_data.mean()) / sample1_data.std() * std1_input + mean1_input
        sample2_data = (sample2_data - sample2_data.mean()) / sample2_data.std() * std2_input + mean2_input

    # Validate
    if sample1_data is None or sample2_data is None:
        st.warning("⚠️ Could not extract data. Please check your selections.")
        return

    if len(sample1_data) < 2 or len(sample2_data) < 2:
        st.error(f"❌ Each sample needs ≥ 2 observations. Got: {label1}={len(sample1_data)}, {label2}={len(sample2_data)}")
        return

    st.info(f"📊 **{label1}**: n = {len(sample1_data)}  |  **{label2}**: n = {len(sample2_data)}")

    # ===== TEST OPTIONS =====
    st.markdown("---")
    st.subheader("3️⃣ Test Options")

    opt_col1, opt_col2, opt_col3 = st.columns(3)

    with opt_col1:
        alternative = st.selectbox(
            "Alternative hypothesis:",
            ["Difference ≠ 0 (two-sided)", "Difference < 0 (less)", "Difference > 0 (greater)"],
            index=0,
            key="ttest_alternative",
            help=(
                "**Two-sided**: Test if means are different (μ₁ ≠ μ₂)\n"
                "**Less**: Test if Sample 1 mean is less than Sample 2\n"
                "**Greater**: Test if Sample 1 mean is greater than Sample 2"
            )
        )
        alt_map = {
            "Difference ≠ 0 (two-sided)": "two-sided",
            "Difference < 0 (less)": "less",
            "Difference > 0 (greater)": "greater"
        }
        alt_value = alt_map[alternative]

    with opt_col2:
        confidence_level = st.selectbox(
            "Confidence level:",
            [0.90, 0.95, 0.99],
            index=1,
            format_func=lambda x: f"{int(x*100)}%",
            key="ttest_conf",
            help="Confidence level for the confidence interval"
        )
        alpha = 1 - confidence_level

    with opt_col3:
        equal_var = st.checkbox(
            "Assume equal variances",
            value=False,
            key="ttest_equal_var",
            help=(
                "**Unchecked (default)**: Welch's t-test — does NOT assume equal variances. "
                "This is Minitab's default and is generally recommended.\n\n"
                "**Checked**: Pooled t-test — assumes both populations have the same variance."
            )
        )

    hyp_diff = st.number_input(
        "Hypothesized difference (μ₁ − μ₂):",
        value=0.0,
        format="%.4f",
        key="ttest_hyp_diff",
        help="Usually 0 (test if means are equal). Set to other values for equivalence/non-inferiority tests."
    )

    # ===== RUN TEST =====
    st.markdown("---")

    if st.button("▶️ Run Two-Sample T-Test", type="primary", use_container_width=True, key="ttest_run"):
        try:
            results = two_sample_ttest(
                sample1_data, sample2_data,
                alternative=alt_value,
                equal_var=equal_var,
                confidence_level=confidence_level,
                hypothesized_difference=hyp_diff
            )
            # Store in session state
            st.session_state.ttest_results = results
            st.session_state.ttest_sample1 = sample1_data
            st.session_state.ttest_sample2 = sample2_data
            st.session_state.ttest_label1 = label1
            st.session_state.ttest_label2 = label2
        except Exception as e:
            st.error(f"❌ Error running test: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    # ===== DISPLAY RESULTS (TABS) =====
    if 'ttest_results' not in st.session_state:
        st.info("👆 Configure your test options and click **Run** to see results.")
        return

    results = st.session_state.ttest_results
    s1 = st.session_state.ttest_sample1
    s2 = st.session_state.ttest_sample2
    lbl1 = st.session_state.ttest_label1
    lbl2 = st.session_state.ttest_label2

    test = results['test_results']
    desc = results['descriptive']
    ci = results['ci']
    eff = results['effect_size']

    st.markdown("---")

    # ===== RESULT TABS =====
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Results",
        "📈 Graphs",
        "✅ Assumptions",
        "💾 Report & Export"
    ])

    # ========== TAB 1: RESULTS ==========
    with tab1:
        st.markdown("## 📋 Test Results")

        # Significance banner
        st.markdown(_significance_badge(test['p_value'], alpha))

        st.markdown("---")

        # Descriptive Statistics table (Minitab-style)
        st.markdown("### Descriptive Statistics")
        desc_df = pd.DataFrame({
            'Sample': [lbl1, lbl2],
            'N': [desc['sample1']['n'], desc['sample2']['n']],
            'Mean': [desc['sample1']['mean'], desc['sample2']['mean']],
            'StDev': [desc['sample1']['std_dev'], desc['sample2']['std_dev']],
            'SE Mean': [desc['sample1']['se_mean'], desc['sample2']['se_mean']]
        })
        st.dataframe(
            desc_df.style.format({
                'Mean': '{:.6f}',
                'StDev': '{:.6f}',
                'SE Mean': '{:.6f}'
            }),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # Estimation for Difference
        st.markdown("### Estimation for Difference")
        ci_pct = int(ci['confidence_level'] * 100)

        est_col1, est_col2 = st.columns(2)
        with est_col1:
            st.metric("Difference (x̄₁ − x̄₂)", f"{test['difference']:.6f}")
            st.metric("SE of Difference", f"{test['se_difference']:.6f}")

        with est_col2:
            if np.isfinite(ci['lower']) and np.isfinite(ci['upper']):
                st.metric(f"{ci_pct}% CI for Difference", f"({ci['lower']:.6f},  {ci['upper']:.6f})")
            elif not np.isfinite(ci['lower']):
                st.metric(f"{ci_pct}% Upper Bound", f"{ci['upper']:.6f}")
            else:
                st.metric(f"{ci_pct}% Lower Bound", f"{ci['lower']:.6f}")

        st.markdown("---")

        # Hypothesis Test
        st.markdown("### Hypothesis Test")

        null_str = f"μ₁ − μ₂ = {test['hypothesized_difference']}"
        st.markdown(f"**H₀:** {null_str}")
        st.markdown(f"**H₁:** {test['alt_description']}")

        test_col1, test_col2, test_col3 = st.columns(3)
        with test_col1:
            st.metric("T-Value", f"{test['t_statistic']:.4f}")
        with test_col2:
            st.metric("DF", f"{test['df']:.2f}")
        with test_col3:
            p_color = "🟢" if test['p_value'] < alpha else "🔴"
            st.metric("P-Value", f"{p_color} {test['p_value']:.6f}")

        st.markdown("---")

        # Effect Size
        st.markdown("### Effect Size")

        d_abs = abs(eff['cohens_d'])
        if d_abs < 0.2:
            interp = "negligible"
        elif d_abs < 0.5:
            interp = "small"
        elif d_abs < 0.8:
            interp = "medium"
        else:
            interp = "large"

        eff_col1, eff_col2, eff_col3 = st.columns(3)
        with eff_col1:
            st.metric("Cohen's d", f"{eff['cohens_d']:.4f}")
        with eff_col2:
            st.metric("Hedges' g", f"{eff['hedges_g']:.4f}")
        with eff_col3:
            st.metric("Interpretation", interp.capitalize())

        # Power (approximate)
        st.markdown("---")
        st.markdown("### Power Analysis")

        try:
            power_val = ttest_power(
                desc['sample1']['n'], desc['sample2']['n'],
                abs(eff['cohens_d']),
                alpha=alpha,
                alternative=alt_value
            )
            pow_col1, pow_col2 = st.columns(2)
            with pow_col1:
                st.metric("Observed Power", f"{power_val:.4f} ({power_val*100:.1f}%)")
            with pow_col2:
                if power_val >= 0.80:
                    st.success("✅ Adequate power (≥ 0.80)")
                else:
                    st.warning(f"⚠️ Low power ({power_val:.2f} < 0.80). Consider larger sample sizes.")
        except Exception:
            st.caption("Power calculation not available for this configuration.")

    # ========== TAB 2: GRAPHS ==========
    with tab2:
        st.markdown("## 📈 Graphical Analysis")

        if not TTEST_PLOTS_AVAILABLE:
            st.error("❌ Plotting module not available.")
        else:
            plot_type = st.selectbox(
                "Select plot:",
                [
                    "📊 Summary (4-in-1)",
                    "🔵 Individual Value Plot",
                    "📦 Boxplot",
                    "📊 Histogram",
                    "📐 Normal Q-Q Plot",
                    "📏 Interval Plot",
                    "📋 Report Card"
                ],
                key="ttest_plot_type"
            )

            try:
                if plot_type == "📊 Summary (4-in-1)":
                    fig = plot_ttest_fourplot(s1, s2, lbl1, lbl2, results, confidence_level)
                elif plot_type == "🔵 Individual Value Plot":
                    fig = plot_individual_value(s1, s2, lbl1, lbl2, results)
                elif plot_type == "📦 Boxplot":
                    fig = plot_comparative_boxplot(s1, s2, lbl1, lbl2)
                elif plot_type == "📊 Histogram":
                    n_bins = st.slider("Number of bins:", 5, 50, 20, key="ttest_bins")
                    fig = plot_histogram_overlay(s1, s2, lbl1, lbl2, n_bins)
                elif plot_type == "📐 Normal Q-Q Plot":
                    fig = plot_qq_by_group(s1, s2, lbl1, lbl2)
                elif plot_type == "📏 Interval Plot":
                    fig = plot_interval_plot(s1, s2, lbl1, lbl2, confidence_level)
                elif plot_type == "📋 Report Card":
                    fig = plot_test_report(results, lbl1, lbl2)

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Plot error: {e}")
                import traceback
                st.code(traceback.format_exc())

    # ========== TAB 3: ASSUMPTIONS ==========
    with tab3:
        st.markdown("## ✅ Assumption Checks")
        st.markdown("""
        The two-sample t-test assumes:
        1. **Independence** — observations are independent
        2. **Normality** — data in each group is approximately normal
        3. **Equal variances** (pooled test only) — both groups have similar spread
        """)

        # --- NORMALITY ---
        st.markdown("---")
        st.markdown("### 🔔 Normality Tests")

        norm_col1, norm_col2 = st.columns(2)

        with norm_col1:
            st.markdown(f"**{lbl1}** (n = {len(s1)})")
            try:
                norm1 = test_normality(s1, lbl1)
                norm1_rows = []
                for test_key in ['shapiro_wilk', 'anderson_darling', 'dagostino_pearson']:
                    if test_key in norm1:
                        t = norm1[test_key]
                        p_val = t.get('p_value', np.nan)
                        if np.isnan(p_val):
                            verdict = t.get('note', 'N/A')
                        elif p_val > 0.05:
                            verdict = "✅ Normal"
                        else:
                            verdict = "⚠️ Non-normal"
                        norm1_rows.append({
                            'Test': t['test_name'],
                            'Statistic': f"{t['statistic']:.4f}" if not np.isnan(t['statistic']) else "N/A",
                            'P-Value': f"{p_val:.4f}" if not np.isnan(p_val) else "N/A",
                            'Verdict': verdict
                        })
                st.dataframe(pd.DataFrame(norm1_rows), use_container_width=True, hide_index=True)
                st.caption(f"Skewness: {norm1['skewness']:.4f} | Kurtosis: {norm1['kurtosis']:.4f}")
            except Exception as e:
                st.warning(f"Could not run normality tests: {e}")

        with norm_col2:
            st.markdown(f"**{lbl2}** (n = {len(s2)})")
            try:
                norm2 = test_normality(s2, lbl2)
                norm2_rows = []
                for test_key in ['shapiro_wilk', 'anderson_darling', 'dagostino_pearson']:
                    if test_key in norm2:
                        t = norm2[test_key]
                        p_val = t.get('p_value', np.nan)
                        if np.isnan(p_val):
                            verdict = t.get('note', 'N/A')
                        elif p_val > 0.05:
                            verdict = "✅ Normal"
                        else:
                            verdict = "⚠️ Non-normal"
                        norm2_rows.append({
                            'Test': t['test_name'],
                            'Statistic': f"{t['statistic']:.4f}" if not np.isnan(t['statistic']) else "N/A",
                            'P-Value': f"{p_val:.4f}" if not np.isnan(p_val) else "N/A",
                            'Verdict': verdict
                        })
                st.dataframe(pd.DataFrame(norm2_rows), use_container_width=True, hide_index=True)
                st.caption(f"Skewness: {norm2['skewness']:.4f} | Kurtosis: {norm2['kurtosis']:.4f}")
            except Exception as e:
                st.warning(f"Could not run normality tests: {e}")

        # Guidance
        st.markdown("""
        > **Interpretation**: If p-value > 0.05, the normality assumption is reasonable.
        > The t-test is robust to moderate departures from normality, especially with larger samples (n > 30).
        """)

        # --- EQUAL VARIANCES ---
        st.markdown("---")
        st.markdown("### 📏 Test for Equal Variances")

        try:
            var_results = test_equal_variances(s1, s2)

            var_rows = []
            for test_key in ['f_test', 'levene', 'bartlett']:
                if test_key in var_results:
                    vt = var_results[test_key]
                    p_val = vt.get('p_value', np.nan)
                    if np.isnan(p_val):
                        verdict = "N/A"
                    elif p_val > 0.05:
                        verdict = "✅ Equal variances"
                    else:
                        verdict = "⚠️ Unequal variances"
                    var_rows.append({
                        'Test': vt['test_name'],
                        'Statistic': f"{vt['statistic']:.4f}" if not np.isnan(vt['statistic']) else "N/A",
                        'P-Value': f"{p_val:.4f}" if not np.isnan(p_val) else "N/A",
                        'Verdict': verdict
                    })

            st.dataframe(pd.DataFrame(var_rows), use_container_width=True, hide_index=True)

            # Show variance ratio
            f_res = var_results.get('f_test', {})
            if 'var1' in f_res and 'var2' in f_res:
                v_col1, v_col2, v_col3 = st.columns(3)
                with v_col1:
                    st.metric(f"Variance ({lbl1})", f"{f_res['var1']:.6f}")
                with v_col2:
                    st.metric(f"Variance ({lbl2})", f"{f_res['var2']:.6f}")
                with v_col3:
                    st.metric("Ratio (s₁²/s₂²)", f"{f_res['ratio']:.4f}")

            # Recommendation
            lev_p = var_results.get('levene', {}).get('p_value', np.nan)
            if not np.isnan(lev_p):
                if lev_p > 0.05:
                    st.success(
                        "✅ **Levene's test suggests equal variances** (p > 0.05). "
                        "Pooled t-test is appropriate, though Welch's test is always safe to use."
                    )
                else:
                    st.warning(
                        "⚠️ **Levene's test suggests unequal variances** (p ≤ 0.05). "
                        "Use **Welch's t-test** (uncheck 'Assume equal variances')."
                    )
                    if equal_var:
                        st.error(
                            "🚨 You selected **pooled t-test** but variances appear unequal. "
                            "Consider switching to Welch's t-test for more reliable results."
                        )
        except Exception as e:
            st.warning(f"Could not run variance tests: {e}")

    # ========== TAB 4: REPORT & EXPORT ==========
    with tab4:
        st.markdown("## 💾 Report & Export")

        # Minitab-style text report
        st.markdown("### 📋 Minitab-Style Report")
        report_text = format_minitab_output(results, lbl1, lbl2)
        st.code(report_text, language="text")

        st.markdown("---")
        st.markdown("### 📥 Download")

        exp_col1, exp_col2, exp_col3 = st.columns(3)

        with exp_col1:
            # CSV - Descriptive stats
            desc_export = pd.DataFrame({
                'Sample': [lbl1, lbl2],
                'N': [desc['sample1']['n'], desc['sample2']['n']],
                'Mean': [desc['sample1']['mean'], desc['sample2']['mean']],
                'StDev': [desc['sample1']['std_dev'], desc['sample2']['std_dev']],
                'SE_Mean': [desc['sample1']['se_mean'], desc['sample2']['se_mean']]
            })

            test_export = pd.DataFrame([{
                'Difference': test['difference'],
                'SE_Difference': test['se_difference'],
                'T_Statistic': test['t_statistic'],
                'DF': test['df'],
                'P_Value': test['p_value'],
                'CI_Lower': ci['lower'],
                'CI_Upper': ci['upper'],
                'Confidence_Level': ci['confidence_level'],
                'Cohens_d': eff['cohens_d'],
                'Hedges_g': eff['hedges_g'],
                'Method': results['method'],
                'Alternative': test['alternative'],
                'Hypothesized_Diff': test['hypothesized_difference']
            }])

            combined_csv = (
                "# Descriptive Statistics\n"
                + desc_export.to_csv(index=False)
                + "\n# Test Results\n"
                + test_export.to_csv(index=False)
            )

            st.download_button(
                "📥 Results (CSV)",
                data=combined_csv,
                file_name="two_sample_ttest_results.csv",
                mime="text/csv",
                key="ttest_csv"
            )

        with exp_col2:
            # Excel with multiple sheets
            try:
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    desc_export.to_excel(writer, sheet_name='Descriptive Stats', index=False)
                    test_export.to_excel(writer, sheet_name='Test Results', index=False)

                    # Raw data
                    max_len = max(len(s1), len(s2))
                    raw_df = pd.DataFrame({
                        lbl1: pd.Series(s1),
                        lbl2: pd.Series(s2)
                    })
                    raw_df.to_excel(writer, sheet_name='Raw Data', index=False)

                    # Metadata
                    meta_df = pd.DataFrame({
                        'Property': ['Analysis', 'Method', 'Software', 'Date'],
                        'Value': [
                            'Two-Sample T-Test',
                            results['method'],
                            'ChemometricSolutions',
                            pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                        ]
                    })
                    meta_df.to_excel(writer, sheet_name='Metadata', index=False)

                buffer.seek(0)

                st.download_button(
                    "📥 Results (Excel)",
                    data=buffer.getvalue(),
                    file_name="two_sample_ttest_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="ttest_xlsx"
                )
            except Exception as e:
                st.warning(f"⚠️ Excel export error: {e}")

        with exp_col3:
            # Text report
            st.download_button(
                "📥 Report (TXT)",
                data=report_text,
                file_name="two_sample_ttest_report.txt",
                mime="text/plain",
                key="ttest_txt"
            )
