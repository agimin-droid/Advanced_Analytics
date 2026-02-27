"""
2-Sample t-Test Page - with Schuirmann's TOST Sample Size Calculator
                      + Before/After Capability Comparison
"""

import streamlit as st
import pandas as pd
import numpy as np

try:
    from ttest_utils.ttest import (
        perform_two_sample_ttest,
        plot_minitab_style_comparison,
        calculate_sample_size_tost,
        estimate_pooled_sd,
        perform_capability_analysis,
        plot_capability_histograms,
        plot_hypothesis_test_bar,
        perform_imr_analysis,
        plot_imr_chart,
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


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Minitab-style HTML report panel
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v, decimals=4):
    """Format a number, return '*' if None."""
    if v is None:
        return "*"
    return f"{v:.{decimals}f}"

def _fmt2(v, decimals=2):
    if v is None:
        return "*"
    return f"{v:.{decimals}f}"

def _signed(v, decimals):
    """Format with explicit sign."""
    if v is None:
        return "*"
    return f"{v:+.{decimals}f}"


def _render_capability_report(result: dict, before_label: str, after_label: str):
    """
    Render the full Minitab Before/After Capability Comparison Summary
    Report using Streamlit layout elements and Plotly charts.
    """

    r = result
    usl = r['specs']['usl']
    lsl = r['specs']['lsl']
    target = r['specs']['target']

    # ===================================================================
    # TITLE BAR
    # ===================================================================
    st.markdown(
        f"""
        <div style="background:#1f3864;color:white;padding:10px 18px;
                    border-radius:6px 6px 0 0;margin-bottom:0;">
            <div style="font-size:16px;font-weight:bold;text-align:center;">
                Before/After Capability Comparison for {before_label} vs {after_label}
            </div>
            <div style="font-size:13px;text-align:center;">Summary Report</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ===================================================================
    # ROW 1: Reduction Indicator  |  Customer Requirements
    # ===================================================================
    col_left, col_right = st.columns([1, 1])

    with col_left:
        red_pct = r['reduction_pct']
        b_pct = r['out_of_spec']['before']
        a_pct = r['out_of_spec']['after']
        arrow_colour = "#2e7d32" if red_pct >= 0 else "#c62828"
        arrow_char = "▼" if red_pct >= 0 else "▲"

        st.markdown(
            f"""
            <div style="border:1px solid #ccc;padding:12px;border-radius:4px;
                        background:#f8f9fa;min-height:100px;">
                <div style="font-weight:bold;font-size:13px;margin-bottom:6px;">
                    Reduction in % Out of Spec
                </div>
                <div style="display:flex;align-items:center;gap:12px;">
                    <div style="font-size:36px;color:{arrow_colour};
                                font-weight:bold;line-height:1;">
                        {arrow_char}<br>
                        <span style="font-size:18px;">{abs(red_pct):.0f}%</span>
                    </div>
                    <div style="font-size:12px;color:#333;">
                        % Out of spec was {"reduced" if red_pct >= 0 else "increased"}
                        by {abs(red_pct):.0f}% from {b_pct:.2f}%
                        to {a_pct:.2f}%.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        lsl_str = f"{lsl}" if lsl is not None else "*"
        tgt_str = f"{target}" if target is not None else "*"
        usl_str = f"{usl}" if usl is not None else "*"

        st.markdown(
            f"""
            <div style="border:1px solid #ccc;padding:12px;border-radius:4px;
                        background:#f8f9fa;min-height:100px;">
                <div style="font-weight:bold;font-size:13px;margin-bottom:8px;">
                    Customer Requirements
                </div>
                <table style="width:100%;text-align:center;font-size:13px;
                              border-collapse:collapse;">
                    <tr style="border-bottom:1px solid #999;">
                        <td style="padding:4px;font-weight:bold;">Lower Spec</td>
                        <td style="padding:4px;font-weight:bold;">Target</td>
                        <td style="padding:4px;font-weight:bold;">Upper Spec</td>
                    </tr>
                    <tr>
                        <td style="padding:6px;font-size:15px;">{lsl_str}</td>
                        <td style="padding:6px;font-size:15px;">{tgt_str}</td>
                        <td style="padding:6px;font-size:15px;">{usl_str}</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ===================================================================
    # ROW 2: Hypothesis Tests  |  Process Characterization Table
    # ===================================================================
    col_tests, col_table = st.columns([1, 1])

    with col_tests:
        # -- Variance test --
        fig_f = plot_hypothesis_test_bar(
            r['tests']['f_test']['p_value'],
            "Was the process standard deviation reduced?",
        )
        st.plotly_chart(fig_f, use_container_width=True, key="cap_f_chart")

        # -- Mean test --
        fig_t = plot_hypothesis_test_bar(
            r['tests']['t_test']['p_value'],
            "Did the process mean change?",
        )
        st.plotly_chart(fig_t, use_container_width=True, key="cap_t_chart")

    with col_table:
        # Build the Process Characterization + Capability table as HTML
        b = r['before']
        a = r['after']
        c = r['change']
        cap = r['capability']
        zbench = r['zbench']
        oos = r['out_of_spec']
        ppm = r['ppm']

        table_html = f"""
        <div style="border:1px solid #ccc;border-radius:4px;padding:8px;background:#f8f9fa;">
        <div style="font-weight:bold;font-size:13px;margin-bottom:6px;
                    border-bottom:2px solid #1f3864;padding-bottom:4px;">
            Process Characterization
        </div>
        <table style="width:100%;font-size:12px;border-collapse:collapse;">
            <tr style="background:#d6dce4;font-weight:bold;">
                <td style="padding:4px 6px;">Statistics</td>
                <td style="padding:4px 6px;text-align:right;">Before</td>
                <td style="padding:4px 6px;text-align:right;">After</td>
                <td style="padding:4px 6px;text-align:right;">Change</td>
            </tr>
            <tr>
                <td style="padding:3px 6px;">Mean</td>
                <td style="padding:3px 6px;text-align:right;">{_fmt(b['mean'])}</td>
                <td style="padding:3px 6px;text-align:right;">{_fmt(a['mean'])}</td>
                <td style="padding:3px 6px;text-align:right;">{_signed(c['mean'], 4)}</td>
            </tr>
            <tr style="background:#f0f0f0;">
                <td style="padding:3px 6px;">StDev(overall)</td>
                <td style="padding:3px 6px;text-align:right;">{_fmt(b['std'])}</td>
                <td style="padding:3px 6px;text-align:right;">{_fmt(a['std'])}</td>
                <td style="padding:3px 6px;text-align:right;">{_signed(c['std'], 5)}</td>
            </tr>
        </table>

        <div style="font-weight:bold;font-size:12px;margin:10px 0 4px 0;
                    border-bottom:1px solid #999;padding-bottom:3px;">
            Actual (overall) capability
        </div>
        <table style="width:100%;font-size:12px;border-collapse:collapse;">
            <tr style="background:#f0f0f0;">
                <td style="padding:3px 6px;">Pp</td>
                <td style="padding:3px 6px;text-align:right;">{_fmt2(cap['before_pp'])}</td>
                <td style="padding:3px 6px;text-align:right;">{_fmt2(cap['after_pp'])}</td>
                <td style="padding:3px 6px;text-align:right;">{_signed(cap['change_pp'], 2) if cap['change_pp'] is not None else '*'}</td>
            </tr>
            <tr>
                <td style="padding:3px 6px;">Ppk</td>
                <td style="padding:3px 6px;text-align:right;">{_fmt2(cap['before_ppk'])}</td>
                <td style="padding:3px 6px;text-align:right;">{_fmt2(cap['after_ppk'])}</td>
                <td style="padding:3px 6px;text-align:right;">{_signed(cap['change_ppk'], 2) if cap['change_ppk'] is not None else '*'}</td>
            </tr>
            <tr style="background:#f0f0f0;">
                <td style="padding:3px 6px;">Z.Bench</td>
                <td style="padding:3px 6px;text-align:right;">{_fmt2(zbench['before'])}</td>
                <td style="padding:3px 6px;text-align:right;">{_fmt2(zbench['after'])}</td>
                <td style="padding:3px 6px;text-align:right;">{_signed(zbench['change'], 2)}</td>
            </tr>
            <tr>
                <td style="padding:3px 6px;">% Out of spec</td>
                <td style="padding:3px 6px;text-align:right;">{_fmt2(oos['before'])}</td>
                <td style="padding:3px 6px;text-align:right;">{_fmt2(oos['after'])}</td>
                <td style="padding:3px 6px;text-align:right;">{_signed(oos['change'], 2)}</td>
            </tr>
            <tr style="background:#f0f0f0;">
                <td style="padding:3px 6px;">PPM (DPMO)</td>
                <td style="padding:3px 6px;text-align:right;">{int(round(ppm['before']))}</td>
                <td style="padding:3px 6px;text-align:right;">{int(round(ppm['after']))}</td>
                <td style="padding:3px 6px;text-align:right;">{int(round(ppm['change'])):+d}</td>
            </tr>
        </table>
        </div>
        """
        st.markdown(table_html, unsafe_allow_html=True)

    # ===================================================================
    # ROW 3: Histograms  |  Comments
    # ===================================================================
    col_hist, col_comments = st.columns([1, 1])

    with col_hist:
        # Determine appropriate subtitle
        specs_used = []
        if lsl is not None:
            specs_used.append("above the lower limit")
        if usl is not None:
            specs_used.append("below the upper limit")
        subtitle = "Are the data " + (" and ".join(specs_used)) + "?"

        st.markdown(
            f"""
            <div style="font-weight:bold;font-size:13px;margin-bottom:2px;">
                Actual (Overall) Capability
            </div>
            <div style="font-size:11px;color:#555;margin-bottom:4px;">
                {subtitle}
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig_hist = plot_capability_histograms(result, before_label, after_label)
        st.plotly_chart(fig_hist, use_container_width=True, key="cap_hist_chart")

    with col_comments:
        st.markdown(
            f"""
            <div style="font-weight:bold;font-size:13px;margin-bottom:6px;">
                Comments
            </div>
            """,
            unsafe_allow_html=True,
        )

        comment_items = r['comments']
        comments_html = f"""
        <div style="border:1px solid #ccc;border-radius:4px;padding:12px;
                    background:#f8f9fa;font-size:12px;min-height:300px;">
            <div style="margin-bottom:8px;">
                <b>Before:</b> {before_label} &nbsp;&nbsp;
                <b>After:</b> {after_label}
            </div>
            <hr style="border:none;border-top:1px solid #ccc;margin:6px 0;">
            <ul style="padding-left:18px;margin:0;">
        """
        for c in comment_items:
            if c.strip():
                comments_html += f'<li style="margin-bottom:4px;">{c}</li>'
        comments_html += """
            </ul>
        </div>
        """
        st.markdown(comments_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────────────

def show():
    if not TTEST_AVAILABLE:
        st.error("❌ 2-Sample t-Test module is not available.")
        return

    st.markdown("# ⚖️ 2-Sample t-Test")
    st.markdown("**Minitab-style t-test + TOST Sample Size + Capability + Before/After I-MR**")
    st.markdown("---")

    tab_test, tab_size, tab_cap, tab_imr = st.tabs([
        "📊 Perform 2-Sample t-Test",
        "📐 Sample Size Calculation (TOST)",
        "📈 Capability Analysis",
        "🔄 Before and After",
    ])

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
        st.markdown("### Sample Size Calculation - Schuirmann's TOST")

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
The estimated or user-specified standard deviation of the selected response variable. It represents the expected variability in the population and directly influences the required sample size in the Schuirmann's TOST formula.

**Equivalence Margin (δ)**  
The largest absolute difference between the two population means that is still considered practically equivalent. The TOST procedure tests whether the true difference lies within the interval [−δ, +δ].

**Alpha (α)**  
The significance level used for each of the two one-sided t-tests that comprise the TOST procedure. It controls the probability of incorrectly concluding equivalence (commonly set to 0.05).

**Power**  
The desired probability (1 − β) of correctly concluding equivalence when the true difference between the means is zero (or within the equivalence margin). Higher power requires a larger sample size (commonly 0.80 or 0.90).
            """)

    # ====================== TAB 3: CAPABILITY ANALYSIS ======================
    with tab_cap:
        st.markdown("### Before/After Capability Comparison")
        st.markdown("*Minitab-style capability analysis — compares process performance before and after an improvement.*")

        # ---- Dataset selection ----
        if WORKSPACE_AVAILABLE:
            available_datasets = get_workspace_datasets()
            if not available_datasets:
                st.warning("⚠️ No datasets available in workspace.")
                st.stop()
            cap_dataset = st.selectbox(
                "Select dataset:",
                list(available_datasets.keys()),
                key="cap_dataset",
            )
            cap_data = available_datasets[cap_dataset]
        else:
            cap_data = st.session_state.get('current_data')
            cap_dataset = "Current Data"

        if cap_data is None:
            st.warning("⚠️ No data loaded. Please load a dataset first.")
            st.stop()

        st.success(f"✅ Using: **{cap_dataset}**")

        cap_numeric = cap_data.select_dtypes(include=['number']).columns.tolist()
        cap_categorical = [c for c in cap_data.columns if c not in cap_numeric]

        if len(cap_numeric) < 1:
            st.error("Dataset must contain at least one numeric column.")
            st.stop()

        # ---- Data arrangement ----
        cap_arrangement = st.radio(
            "Data arrangement:",
            [
                "Two separate columns (Before / After)",
                "Stacked (Response + Grouping column)",
            ],
            horizontal=True,
            key="cap_arrangement",
        )

        if cap_arrangement.startswith("Two"):
            c1, c2 = st.columns(2)
            with c1:
                before_col = st.selectbox("Before column:", cap_numeric, key="cap_before_col")
            with c2:
                after_col = st.selectbox(
                    "After column:",
                    [c for c in cap_numeric if c != before_col],
                    key="cap_after_col",
                )
            before_label = before_col
            after_label  = after_col
        else:
            c1, c2 = st.columns(2)
            with c1:
                cap_response = st.selectbox("Response variable:", cap_numeric, key="cap_response")
            with c2:
                cap_group = st.selectbox(
                    "Grouping variable (exactly 2 levels):",
                    cap_categorical if cap_categorical else cap_data.columns.tolist(),
                    key="cap_group",
                )

            unique_groups = sorted(cap_data[cap_group].dropna().unique())
            if len(unique_groups) != 2:
                st.error(f"Grouping column must have exactly 2 levels (found {len(unique_groups)}).")
                st.stop()

            c1, c2 = st.columns(2)
            with c1:
                before_group = st.selectbox(
                    "Which level is BEFORE?",
                    unique_groups,
                    key="cap_before_group",
                )
            with c2:
                after_group = [g for g in unique_groups if g != before_group][0]
                st.info(f"AFTER level: **{after_group}**")

            before_label = str(before_group)
            after_label  = str(after_group)

        # ---- Specification limits ----
        st.markdown("#### Customer Requirements (Specification Limits)")

        c1, c2, c3 = st.columns(3)
        with c1:
            use_lsl = st.checkbox("Lower Spec Limit (LSL)", value=False, key="cap_use_lsl")
            lsl_val = st.number_input("LSL value:", value=0.0, format="%.4f", key="cap_lsl",
                                      disabled=not use_lsl)
        with c2:
            use_target = st.checkbox("Target", value=False, key="cap_use_target")
            target_val = st.number_input("Target value:", value=0.0, format="%.4f", key="cap_target",
                                         disabled=not use_target)
        with c3:
            use_usl = st.checkbox("Upper Spec Limit (USL)", value=True, key="cap_use_usl")
            usl_val = st.number_input("USL value:", value=8.0, format="%.4f", key="cap_usl",
                                      disabled=not use_usl)

        cap_alpha = st.slider("Significance level (α) for hypothesis tests:", 0.01, 0.10, 0.05, 0.01,
                              key="cap_alpha")

        if not use_lsl and not use_usl:
            st.error("At least one specification limit (LSL or USL) is required.")
            st.stop()

        # ---- Run ----
        if st.button("📈 Run Capability Analysis", type="primary", use_container_width=True, key="cap_run"):
            try:
                # Extract data arrays
                if cap_arrangement.startswith("Two"):
                    before_arr = pd.to_numeric(cap_data[before_col], errors='coerce').dropna().values
                    after_arr  = pd.to_numeric(cap_data[after_col],  errors='coerce').dropna().values
                else:
                    before_arr = pd.to_numeric(
                        cap_data[cap_data[cap_group] == before_group][cap_response],
                        errors='coerce').dropna().values
                    after_arr = pd.to_numeric(
                        cap_data[cap_data[cap_group] == after_group][cap_response],
                        errors='coerce').dropna().values

                usl = usl_val if use_usl else None
                lsl = lsl_val if use_lsl else None
                target = target_val if use_target else None

                cap_result = perform_capability_analysis(
                    before_arr, after_arr,
                    usl=usl, lsl=lsl, target=target,
                    alpha=cap_alpha,
                )

                # Store for re-rendering
                st.session_state.cap_result = cap_result
                st.session_state.cap_before_label = before_label
                st.session_state.cap_after_label = after_label

            except Exception as e:
                st.error(f"❌ Capability analysis failed: {e}")
                import traceback
                st.code(traceback.format_exc())

        # ---- Render stored result (persists across reruns) ----
        if 'cap_result' in st.session_state:
            st.markdown("---")
            _render_capability_report(
                st.session_state.cap_result,
                st.session_state.cap_before_label,
                st.session_state.cap_after_label,
            )

        # ---- Explanation ----
        with st.expander("📘 Interpretation Guide", expanded=False):
            st.markdown("""
**Process Characterization**  
Compares the mean and standard deviation (overall) of the process before and after improvement. The *Change* column shows the absolute shift.

**Pp (Process Performance)**  
Pp = (USL − LSL) / (6σ). Measures overall process spread relative to the specification width. Only available when both LSL and USL are provided.

**Ppk (Process Performance Index)**  
Ppk = min((USL − μ)/(3σ), (μ − LSL)/(3σ)). Accounts for process centering. Higher values indicate better capability (≥ 1.33 is often the target).

**Z.Bench (Benchmark Z-score)**  
The number of standard deviations between the process centre and the nearest specification limit, expressed as a normal quantile. Directly related to PPM.

**% Out of Spec / PPM (DPMO)**  
The estimated percentage (and parts per million) of output falling outside specification limits, assuming a normal distribution.

**Was the standard deviation reduced?**  
One-sided F-test (H₁: σ_before > σ_after). A small p-value (< α) means the variability was significantly reduced.

**Did the process mean change?**  
Two-sided Welch t-test. A small p-value (< α) means the mean shifted significantly.
            """)

    # ====================== TAB 4: BEFORE AND AFTER (I-MR) ======================
    with tab_imr:
        st.markdown("### Before/After I-MR Chart")
        st.markdown("*Minitab-style Individuals & Moving Range control chart — compares process behaviour across two stages.*")

        # ---- Dataset selection ----
        if WORKSPACE_AVAILABLE:
            imr_datasets = get_workspace_datasets()
            if not imr_datasets:
                st.warning("⚠️ No datasets available in workspace.")
                st.stop()
            imr_ds_name = st.selectbox(
                "Select dataset:",
                list(imr_datasets.keys()),
                key="imr_dataset",
            )
            imr_data = imr_datasets[imr_ds_name]
        else:
            imr_data = st.session_state.get('current_data')
            imr_ds_name = "Current Data"

        if imr_data is None:
            st.warning("⚠️ No data loaded. Please load a dataset first.")
            st.stop()

        st.success(f"✅ Using: **{imr_ds_name}**")

        imr_numeric = imr_data.select_dtypes(include=['number']).columns.tolist()
        imr_categorical = [c for c in imr_data.columns if c not in imr_numeric]

        if len(imr_numeric) < 1:
            st.error("Dataset must contain at least one numeric column.")
            st.stop()

        # ---- Data arrangement ----
        imr_arrangement = st.radio(
            "Data arrangement:",
            [
                "Stacked (Response + Grouping column)",
                "Two separate columns (Before / After)",
            ],
            horizontal=True,
            key="imr_arrangement",
        )

        if imr_arrangement.startswith("Stacked"):
            c1, c2 = st.columns(2)
            with c1:
                imr_response = st.selectbox("Response variable:", imr_numeric, key="imr_response")
            with c2:
                imr_group = st.selectbox(
                    "Grouping variable (exactly 2 levels):",
                    imr_categorical if imr_categorical else imr_data.columns.tolist(),
                    key="imr_group",
                )

            unique_groups = sorted(imr_data[imr_group].dropna().unique())
            if len(unique_groups) != 2:
                st.error(f"Grouping column must have exactly 2 levels (found {len(unique_groups)}).")
                st.stop()

            c1, c2 = st.columns(2)
            with c1:
                imr_before_group = st.selectbox(
                    "Which level is BEFORE?",
                    unique_groups,
                    key="imr_before_group",
                )
            with c2:
                imr_after_group = [g for g in unique_groups if g != imr_before_group][0]
                st.info(f"AFTER level: **{imr_after_group}**")

            imr_before_label = str(imr_before_group)
            imr_after_label  = str(imr_after_group)
        else:
            c1, c2 = st.columns(2)
            with c1:
                imr_before_col = st.selectbox("Before column:", imr_numeric, key="imr_before_col")
            with c2:
                imr_after_col = st.selectbox(
                    "After column:",
                    [c for c in imr_numeric if c != imr_before_col],
                    key="imr_after_col",
                )
            imr_before_label = imr_before_col
            imr_after_label  = imr_after_col

        imr_alpha = st.slider(
            "Significance level (α) for hypothesis tests:",
            0.01, 0.10, 0.05, 0.01,
            key="imr_alpha",
        )

        # ---- Run ----
        if st.button("🔄 Run Before/After I-MR Analysis", type="primary",
                      use_container_width=True, key="imr_run"):
            try:
                # Extract data arrays
                if imr_arrangement.startswith("Stacked"):
                    before_arr = pd.to_numeric(
                        imr_data[imr_data[imr_group] == imr_before_group][imr_response],
                        errors='coerce').dropna().values
                    after_arr = pd.to_numeric(
                        imr_data[imr_data[imr_group] == imr_after_group][imr_response],
                        errors='coerce').dropna().values
                    resp_label = imr_response
                else:
                    before_arr = pd.to_numeric(imr_data[imr_before_col], errors='coerce').dropna().values
                    after_arr  = pd.to_numeric(imr_data[imr_after_col],  errors='coerce').dropna().values
                    resp_label = "Individual Value"

                imr_result = perform_imr_analysis(before_arr, after_arr, alpha=imr_alpha)

                st.session_state.imr_result = imr_result
                st.session_state.imr_before_label = imr_before_label
                st.session_state.imr_after_label  = imr_after_label
                st.session_state.imr_resp_label   = resp_label

            except Exception as e:
                st.error(f"❌ I-MR analysis failed: {e}")
                import traceback
                st.code(traceback.format_exc())

        # ---- Render stored result ----
        if 'imr_result' in st.session_state:
            r = st.session_state.imr_result
            bl = st.session_state.imr_before_label
            al = st.session_state.imr_after_label
            rl = st.session_state.imr_resp_label

            st.markdown("---")

            # ============================================================
            # TITLE BAR
            # ============================================================
            st.markdown(
                f"""
                <div style="background:#1f3864;color:white;padding:10px 18px;
                            border-radius:6px 6px 0 0;margin-bottom:0;">
                    <div style="font-size:16px;font-weight:bold;text-align:center;">
                        Before/After I-MR Chart of {rl} by {bl} / {al}
                    </div>
                    <div style="font-size:13px;text-align:center;">Summary Report</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ============================================================
            # ROW 1: Hypothesis test bars  |  Comments
            # ============================================================
            col_tests, col_comments = st.columns([1, 1])

            with col_tests:
                fig_f = plot_hypothesis_test_bar(
                    r['tests']['f_test']['p_value'],
                    "Was the process standard deviation reduced?",
                    alpha=imr_alpha if 'imr_alpha' in dir() else 0.05,
                )
                st.plotly_chart(fig_f, use_container_width=True, key="imr_f_chart")

                fig_t = plot_hypothesis_test_bar(
                    r['tests']['t_test']['p_value'],
                    "Did the process mean change?",
                    alpha=imr_alpha if 'imr_alpha' in dir() else 0.05,
                )
                st.plotly_chart(fig_t, use_container_width=True, key="imr_t_chart")

            with col_comments:
                st.markdown(
                    """
                    <div style="font-weight:bold;font-size:13px;margin-bottom:6px;">
                        Comments
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                comments_html = """
                <div style="border:1px solid #ccc;border-radius:4px;padding:12px;
                            background:#f8f9fa;font-size:12px;min-height:180px;">
                    <ul style="padding-left:18px;margin:0;">
                """
                for c in r['comments']:
                    if c.strip():
                        comments_html += f'<li style="margin-bottom:5px;">{c}</li>'
                comments_html += "</ul></div>"
                st.markdown(comments_html, unsafe_allow_html=True)

            # ============================================================
            # ROW 2: I-MR Chart (full width)
            # ============================================================
            fig_imr = plot_imr_chart(r, bl, al, rl)
            st.plotly_chart(fig_imr, use_container_width=True, key="imr_chart")

            # ============================================================
            # ROW 3: Stage Statistics Table
            # ============================================================
            b = r['before']
            a = r['after']

            table_html = f"""
            <div style="border:1px solid #ccc;border-radius:4px;padding:10px;
                        background:#f8f9fa;">
            <table style="width:100%;font-size:13px;border-collapse:collapse;
                          text-align:center;">
                <tr style="background:#d6dce4;font-weight:bold;">
                    <td style="padding:6px;text-align:left;">Stage</td>
                    <td style="padding:6px;">N</td>
                    <td style="padding:6px;">Mean</td>
                    <td style="padding:6px;">StDev(Within)</td>
                    <td style="padding:6px;">StDev(Overall)</td>
                </tr>
                <tr>
                    <td style="padding:5px 6px;text-align:left;font-weight:bold;">{bl}</td>
                    <td style="padding:5px 6px;">{b['n']}</td>
                    <td style="padding:5px 6px;">{b['mean']:.3f}</td>
                    <td style="padding:5px 6px;">{b['sd_within']:.4f}</td>
                    <td style="padding:5px 6px;">{b['sd_overall']:.4f}</td>
                </tr>
                <tr style="background:#f0f0f0;">
                    <td style="padding:5px 6px;text-align:left;font-weight:bold;">{al}</td>
                    <td style="padding:5px 6px;">{a['n']}</td>
                    <td style="padding:5px 6px;">{a['mean']:.3f}</td>
                    <td style="padding:5px 6px;">{a['sd_within']:.4f}</td>
                    <td style="padding:5px 6px;">{a['sd_overall']:.4f}</td>
                </tr>
            </table>
            <div style="font-size:11px;color:#666;text-align:right;
                        margin-top:6px;font-style:italic;">
                Control limits use StDev(Within)
            </div>
            </div>
            """
            st.markdown(table_html, unsafe_allow_html=True)

        # ---- Explanation ----
        with st.expander("📘 Interpretation Guide", expanded=False):
            st.markdown("""
**I-MR Chart (Individuals & Moving Range)**  
The I-MR chart is used when data are collected one observation at a time (no subgroups). It is the standard Minitab control chart for continuous, individual measurements.

**Individuals Chart (top)**  
Plots each observation in time order. The centre line (X̄) and control limits (UCL, LCL) are calculated from the **After** stage, which represents the improved or current process. Points from the Before stage falling outside these limits confirm the process has changed.

**Moving Range Chart (bottom)**  
Plots |x_i − x_{i-1}| — the absolute difference between consecutive observations — as a measure of short-term variation. Control limits use MR̄ and D₄ = 3.267 for span 2.

**StDev(Within) vs StDev(Overall)**  
StDev(Within) = MR̄ / d₂ (d₂ = 1.128). This estimates the inherent short-term process variation and is used for control limits. StDev(Overall) is the ordinary sample standard deviation, which also includes any shifts or drifts.

**Was the standard deviation reduced?**  
One-sided F-test (H₁: σ_before > σ_after). A small p-value (< α) means variability was significantly reduced.

**Did the process mean change?**  
Two-sided Welch t-test. A small p-value (< α) means the mean shifted. The comments indicate the direction — make sure it represents an improvement.
            """)

    st.caption("© 2026 Roquette Advanced Analytics")
