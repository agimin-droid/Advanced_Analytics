"""
2-Sample t-Test Page - with Schuirmann’s TOST Sample Size Calculator
"""

import streamlit as st
import pandas as pd

try:
    from ttest_utils.ttest import (
        perform_two_sample_ttest,
        plot_minitab_style_comparison,
        calculate_sample_size_tost,
        estimate_pooled_sd   # kept for compatibility
    )
    TTEST_AVAILABLE = True
except ImportError as e:
    TTEST_AVAILABLE = False
    st.error(f"❌ ttest_utils import failed: {e}")

try:
    from workspace_utils import get_workspace_datasets
    WORKSPACE_AVAILABLE = True
except ImportError:
    WORKSPACE_AVAILABLE = False


def show():
    if not TTEST_AVAILABLE:
        st.error("❌ 2-Sample t-Test module is not available.")
        return

    st.markdown("# ⚖️ 2-Sample t-Test")
    st.markdown("**Minitab-style t-test + Schuirmann’s TOST Sample Size Calculator**")
    st.markdown("---")

    tab_test, tab_size = st.tabs(["📊 Perform 2-Sample t-Test", "📐 Sample Size Calculation (TOST)"])

    # ====================== TAB 1: PERFORM TEST ======================
    with tab_test:
        st.markdown("### Perform 2-Sample t-Test")

        if WORKSPACE_AVAILABLE:
            available_datasets = get_workspace_datasets()
            if not available_datasets:
                st.warning("⚠️ No datasets available.")
                return
            dataset_name = st.selectbox("Select dataset:", list(available_datasets.keys()), key="ttest_dataset")
            data = available_datasets[dataset_name]
        else:
            data = st.session_state.get('current_data')
            dataset_name = "Current Data"

        st.success(f"✅ Using: **{dataset_name}**")

        numeric_vars = data.select_dtypes(include=['number']).columns.tolist()
        categorical_vars = [col for col in data.columns if col not in numeric_vars]

        data_format = st.radio("Data arrangement:", ["Stacked (Response + Grouping column)", "Unstacked (Two separate columns)"], horizontal=True, key="ttest_format")

        if data_format.startswith("Stacked"):
            c1, c2 = st.columns(2)
            with c1: response_col = st.selectbox("Response variable:", numeric_vars, key="ttest_resp")
            with c2: group_col = st.selectbox("Grouping variable (exactly 2 levels):", categorical_vars, key="ttest_group")
        else:
            c1, c2 = st.columns(2)
            with c1: col1_name = st.selectbox("Sample 1 column:", numeric_vars, key="ttest_col1")
            with c2: col2_name = st.selectbox("Sample 2 column:", [c for c in numeric_vars if c != col1_name], key="ttest_col2")
            response_col = group_col = None

        c1, c2, c3 = st.columns(3)
        with c1: equal_var = st.checkbox("Assume equal variances", value=True)
        with c2: conf_level = st.slider("Confidence level (%)", 80, 99, 95) / 100.0
        with c3: 
            alt = st.selectbox("Alternative hypothesis", ["two-sided", "greater than", "less than"])
            alternative = {"two-sided": "two-sided", "greater than": "greater", "less than": "less"}[alt]

        if st.button("🚀 Perform 2-Sample t-Test", type="primary", use_container_width=True):
            try:
                if data_format.startswith("Stacked"):
                    result = perform_two_sample_ttest(data, response_col=response_col, group_col=group_col, equal_var=equal_var, conf_level=conf_level, alternative=alternative)
                    fig = plot_minitab_style_comparison(data, response_col, group_col)
                else:
                    result = perform_two_sample_ttest(data, col1=col1_name, col2=col2_name, equal_var=equal_var, conf_level=conf_level, alternative=alternative)
                    fig = plot_minitab_style_comparison(data, col1_name)

                st.success(result['conclusion'])
                st.subheader("Descriptive Statistics")
                st.dataframe(pd.DataFrame(result['descriptive']).T.style.format("{:.4f}"), use_container_width=True)
                st.subheader("Test Results")
                diff = result['difference']
                test = result['test']
                ca, cb = st.columns(2)
                with ca:
                    st.metric("Difference", f"{diff['estimate']:.4f}")
                    st.metric(f"{diff['confidence']:.0f}% CI", f"({diff['ci_lower']:.4f}, {diff['ci_upper']:.4f})")
                with cb:
                    st.metric("t-Value", f"{test['t_value']:.4f}")
                    st.metric("p-Value", f"{test['p_value']:.4f}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

    # ====================== TAB 2: SAMPLE SIZE CALCULATION ======================
    with tab_size:
        st.markdown("### Sample Size Calculation - Schuirmann’s TOST")

        # ------------------- Standard Deviation Estimation -------------------
        st.markdown("#### Estimate Standard Deviation from Dataset")

        if not WORKSPACE_AVAILABLE:
            st.error("Workspace utilities not available.")
        else:
            available_datasets = get_workspace_datasets()
            if not available_datasets:
                st.warning("No datasets available in workspace.")
            else:
                dataset_name = st.selectbox(
                    "Select dataset:",
                    list(available_datasets.keys()),
                    key="tost_sd_dataset"
                )
                data = available_datasets[dataset_name]

                numeric_vars = data.select_dtypes(include=["number"]).columns.tolist()

                response_col = st.selectbox(
                    "Response variable (the parameter):",
                    numeric_vars,
                    key="tost_sd_response"
                )

                if st.button(
                    "🔄 Calculate Standard Deviation from Selected Parameter",
                    type="primary",
                    use_container_width=True,
                    key="calc_sd_btn"
                ):
                    try:
                        values = pd.to_numeric(data[response_col], errors='coerce').dropna()
                        if len(values) < 2:
                            raise ValueError("Not enough data points (minimum 2 required).")
                        sigma = values.std(ddof=1)
                        st.session_state.tost_current_sigma = sigma
                        st.session_state.tost_sd_source = f"{dataset_name} — {response_col}"
                        st.success(f"✅ Standard Deviation calculated: **{sigma:.4f}**")
                    except Exception as e:
                        st.error(f"Calculation failed: {e}")

        # ------------------- Persistent Sigma -------------------
        if "tost_current_sigma" not in st.session_state:
            st.session_state.tost_current_sigma = 1.0

        # ------------------- Parameter Inputs (Requested Layout) -------------------
        col_sigma, col_delta = st.columns(2)
        with col_sigma:
            sigma = st.number_input(
                "Manual Standard Deviation",
                value=st.session_state.tost_current_sigma,
                min_value=0.01,
                step=0.0001,
                format="%.4f",
                key="tost_sigma_override"
            )
            st.session_state.tost_current_sigma = sigma

        with col_delta:
            delta = st.number_input(
                "Equivalence Margin (δ)",
                value=0.5,
                min_value=0.01,
                step=0.01,
                format="%.3f",
                key="tost_delta"
            )

        col_alpha, col_power = st.columns(2)
        with col_alpha:
            alpha = st.slider("Alpha", 0.01, 0.10, 0.05, step=0.01, key="tost_alpha")
        with col_power:
            power = st.slider("Power", 0.70, 0.99, 0.80, step=0.01, key="tost_power")

        # Show source of last calculation
        if "tost_sd_source" in st.session_state:
            st.success(f"**Last calculated SD source:** {st.session_state.tost_sd_source}")
        else:
            st.info("Standard deviation has not been calculated yet. Use the button above to populate the field.")

        # Calculate sample size
        if st.button(
            "📐 Calculate Required Sample Size",
            type="primary",
            use_container_width=True,
            key="tost_calc_size_btn"
        ):
            if sigma <= 0:
                st.error("Standard deviation must be positive.")
            else:
                result = calculate_sample_size_tost(sigma, delta, alpha, power)
                st.success(f"**Sample size per group: {result['n_per_group']}**")
                st.success(f"**Total samples: {result['total_samples']}**")
                st.caption(
                    f"Parameters used → σ = {sigma:.4f} | δ = {delta:.3f} | α = {alpha} | Power = {power}"
                )

        # ------------------- Expandable Explanations (at the bottom) -------------------
        with st.expander("📘 Explanation of Parameters", expanded=False):
            st.markdown("""
**Standard Deviation (σ)**  
The estimated or user-specified standard deviation of the selected response variable. It represents the expected variability in the population and directly influences the required sample size in the Schuirmann’s TOST formula.

**Equivalence Margin (δ)**  
The largest absolute difference between the two population means that is still considered practically equivalent. The TOST procedure tests whether the true difference lies within the interval [−δ, +δ].

**Alpha (α)**  
The significance level used for each of the two one-sided t-tests that comprise the TOST procedure. It controls the probability of incorrectly concluding equivalence (commonly set to 0.05).

**Power**  
The desired probability (1 − β) of correctly concluding equivalence when the true difference between the means is zero (or within the equivalence margin). Higher power requires a larger sample size (commonly 0.80 or 0.90).
            """)

    st.caption("© 2026 Roquette Advanced Analytics")