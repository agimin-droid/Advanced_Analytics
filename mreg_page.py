"""
Multiple Regression Page – Minitab-Style Reports
Supports: OLS, Ridge, Lasso, ElasticNet, PLS
"""

import streamlit as st
import pandas as pd
import numpy as np
import textwrap

try:
    from mreg_utils.mreg import (
        generate_extended_features,
        compute_ols_statistics,
        forward_stepwise,
        fit_model,
        compute_incremental_impact,
        compute_x_regressed_on_others,
        detect_unusual_data,
        check_residual_normality,
        generate_report_card,
        format_equation,
        plot_model_building_sequence,
        plot_incremental_impact,
        plot_x_regressed_on_others,
        plot_pvalue_bar,
        plot_scatter_panels,
        plot_diagnostic_report,
        export_results_to_excel,
    )
    MREG_AVAILABLE = True
except ImportError as e:
    MREG_AVAILABLE = False
    _import_err = str(e)

try:
    from workspace_utils import get_workspace_datasets
    WORKSPACE_AVAILABLE = True
except ImportError:
    WORKSPACE_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Helper: dedent HTML so Streamlit doesn't treat indentation as code blocks
# ─────────────────────────────────────────────────────────────────────────────

def _html(raw: str) -> str:
    """Strip leading whitespace from every line to prevent markdown code blocks."""
    return textwrap.dedent(raw).strip()


# ─────────────────────────────────────────────────────────────────────────────
# HTML rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

_STATUS_ICONS = {
    "ok":      '<span style="display:inline-block;width:28px;height:28px;border-radius:4px;background:#2e7d32;color:white;text-align:center;line-height:28px;font-size:18px;">&#10003;</span>',
    "warning": '<span style="display:inline-block;width:28px;height:28px;background:#f9a825;border-radius:4px;color:#333;text-align:center;line-height:28px;font-size:16px;font-weight:bold;">&#9888;</span>',
    "info":    '<span style="display:inline-block;width:28px;height:28px;border-radius:50%;background:#1565c0;color:white;text-align:center;line-height:28px;font-size:16px;font-weight:bold;">i</span>',
}


def _render_report_card(checks, response_name):
    """Render Minitab-style report card."""
    st.markdown(_html(f"""
    <div style="background:#1f3864;color:white;padding:10px 18px;border-radius:6px 6px 0 0;text-align:center;">
    <div style="font-size:16px;font-weight:bold;">Multiple Regression for {response_name}</div>
    <div style="font-size:13px;">Report Card</div>
    </div>
    """), unsafe_allow_html=True)

    # Build table rows
    rows_html = ""
    for c in checks:
        icon = _STATUS_ICONS.get(c["status"], _STATUS_ICONS["info"])
        desc = c["description"].replace("\n", "<br>").replace("• ", "&bull; ")
        rows_html += (
            f'<tr style="border-top:1px solid #ccc;">'
            f'<td style="padding:10px 8px;vertical-align:top;font-weight:bold;">{c["check"]}</td>'
            f'<td style="padding:10px 8px;vertical-align:top;text-align:center;">{icon}</td>'
            f'<td style="padding:10px 8px;vertical-align:top;line-height:1.5;">{desc}</td>'
            f'</tr>'
        )

    table_html = (
        '<div style="border:1px solid #ccc;border-top:none;padding:0;background:#f8f9fa;">'
        '<table style="width:100%;font-size:13px;border-collapse:collapse;">'
        '<tr style="background:#d6dce4;font-weight:bold;">'
        '<td style="padding:8px;width:120px;">Check</td>'
        '<td style="padding:8px;width:50px;text-align:center;">Status</td>'
        '<td style="padding:8px;">Description</td>'
        '</tr>'
        f'{rows_html}'
        '</table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


def _render_rsquared_bar(r_sq_pct):
    """Render gradient R² bar (Low → High) like Minitab."""
    position = max(0, min(100, r_sq_pct))
    st.markdown(_html(f"""
    <div style="margin:8px 0;">
    <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:2px;">
    <span>0%</span><span>100%</span>
    </div>
    <div style="position:relative;height:28px;border-radius:4px;background:linear-gradient(to right, #d4a017, #6aad5a, #2e7d32);border:1px solid #999;">
    <div style="position:absolute;left:{position}%;top:-2px;transform:translateX(-50%);width:3px;height:32px;background:#333;"></div>
    <div style="position:absolute;left:{position}%;top:32px;transform:translateX(-50%);font-size:11px;font-weight:bold;white-space:nowrap;background:#f8f9fa;padding:1px 6px;border-radius:3px;border:1px solid #999;">R-sq = {r_sq_pct:.2f}%</div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:12px;font-weight:bold;margin-top:2px;">
    <span style="color:#b8860b;">Low</span>
    <span style="color:#2e7d32;">High</span>
    </div>
    </div>
    """), unsafe_allow_html=True)


def _render_step_table(steps):
    """Render the Model Building Sequence table as HTML string."""
    rows = ""
    for s in steps:
        sp = f"{s['step_p']:.3f}" if s["step_p"] is not None else "-"
        fp = f"{s['final_p']:.3f}" if s["final_p"] is not None else "-"
        rows += (
            f'<tr>'
            f'<td style="padding:4px 6px;">{s["step"]}</td>'
            f'<td style="padding:4px 6px;">{s["change"]}</td>'
            f'<td style="padding:4px 6px;text-align:right;">{sp}</td>'
            f'<td style="padding:4px 6px;text-align:right;">{fp}</td>'
            f'</tr>'
        )

    return (
        '<table style="font-size:12px;border-collapse:collapse;width:100%;">'
        '<tr style="font-weight:bold;border-bottom:2px solid #999;">'
        '<td style="padding:4px 6px;">Step</td>'
        '<td style="padding:4px 6px;">Change</td>'
        '<td style="padding:4px 6px;text-align:right;">Step P</td>'
        '<td style="padding:4px 6px;text-align:right;">Final P</td>'
        '</tr>'
        f'{rows}'
        '</table>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────────────

def show():
    if not MREG_AVAILABLE:
        st.error(f"❌ mreg_utils import failed: {_import_err}")
        return

    st.markdown("# 📈 Multiple Regression")
    st.markdown("**Minitab-style regression analysis — OLS, Ridge, Lasso, ElasticNet, PLS**")
    st.markdown("---")

    # ==================================================================
    # DATA & VARIABLE SELECTION
    # ==================================================================
    st.markdown("### Data & Variable Selection")

    if WORKSPACE_AVAILABLE:
        available_datasets = get_workspace_datasets()
        if not available_datasets:
            st.warning("⚠️ No datasets available in workspace.")
            return
        ds_name = st.selectbox("Select dataset:", list(available_datasets.keys()),
                               key="mreg_dataset")
        data = available_datasets[ds_name]
    else:
        data = st.session_state.get("current_data")
        ds_name = st.session_state.get("current_dataset", "Current Data")

    if data is None:
        st.warning("⚠️ No data loaded. Please load a dataset first.")
        return

    st.success(f"✅ Using: **{ds_name}** ({data.shape[0]} rows × {data.shape[1]} columns)")

    numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
    all_cols = data.columns.tolist()
    categorical_cols = [c for c in all_cols if c not in numeric_cols]

    # Minitab-style selectors
    col_list, col_sel = st.columns([1, 3])

    with col_list:
        col_listing = "<br>".join([f"C{i+1} &nbsp; {c}" for i, c in enumerate(all_cols)])
        st.markdown(
            f'<div style="background:#e8e8e8;padding:8px;border:1px solid #999;'
            f'border-radius:4px;font-family:monospace;font-size:12px;'
            f'max-height:220px;overflow-y:auto;">{col_listing}</div>',
            unsafe_allow_html=True,
        )

    with col_sel:
        response_var = st.selectbox("**Response:**", numeric_cols, key="mreg_response")

        available_predictors = [c for c in numeric_cols if c != response_var]
        continuous_preds = st.multiselect(
            "**Continuous predictors:**",
            available_predictors,
            default=available_predictors[:min(4, len(available_predictors))],
            key="mreg_continuous",
        )

        categorical_preds = st.multiselect(
            "**Categorical predictors:**",
            categorical_cols,
            key="mreg_categorical",
        )

    if not continuous_preds and not categorical_preds:
        st.info("Select at least one predictor to continue.")
        return

    # ==================================================================
    # MODEL CONFIGURATION
    # ==================================================================
    st.markdown("### Model Configuration")

    col_model, col_feat, col_params = st.columns(3)

    with col_model:
        model_type = st.selectbox(
            "Model type:",
            ["Multiple Linear Regression (OLS)",
             "Ridge Regression",
             "Lasso Regression",
             "ElasticNet",
             "PLS Regression"],
            key="mreg_model_type",
        )
        model_key = {
            "Multiple Linear Regression (OLS)": "OLS",
            "Ridge Regression": "Ridge",
            "Lasso Regression": "Lasso",
            "ElasticNet": "ElasticNet",
            "PLS Regression": "PLS",
        }[model_type]

    with col_feat:
        include_squares = st.checkbox("Include squared terms (X²)", value=True,
                                      key="mreg_squares")
        include_interactions = st.checkbox("Include interaction terms (X₁·X₂)",
                                          value=True, key="mreg_interactions")

    with col_params:
        if model_key in ("Ridge", "Lasso", "ElasticNet"):
            reg_alpha = st.number_input("Regularization α:", value=1.0,
                                        min_value=0.001, step=0.1,
                                        format="%.3f", key="mreg_alpha")
        else:
            reg_alpha = 1.0

        if model_key == "ElasticNet":
            l1_ratio = st.slider("L1 ratio:", 0.0, 1.0, 0.5, 0.05,
                                 key="mreg_l1ratio")
        else:
            l1_ratio = 0.5

        if model_key == "PLS":
            max_comp = min(len(continuous_preds), data.shape[0] - 1, 20)
            n_components = st.slider("PLS components:", 1, max(1, max_comp),
                                     min(2, max_comp), key="mreg_ncomp")
        else:
            n_components = 2

    # ==================================================================
    # RUN
    # ==================================================================
    if st.button("🚀 Run Multiple Regression", type="primary",
                 use_container_width=True, key="mreg_run"):
        with st.spinner("Running regression analysis..."):
            try:
                # --- prepare data ------------------------------------------------
                y = pd.to_numeric(data[response_var], errors="coerce").values
                X_cont_df = data[continuous_preds].apply(pd.to_numeric, errors="coerce")

                # Handle categorical predictors (dummy encoding)
                cat_dummy_names = []
                if categorical_preds:
                    cat_dummies = pd.get_dummies(data[categorical_preds],
                                                 drop_first=True, dtype=float)
                    cat_dummy_names = cat_dummies.columns.tolist()
                else:
                    cat_dummies = pd.DataFrame()

                # Combine for NaN mask
                X_all_df = pd.concat([X_cont_df, cat_dummies], axis=1)
                mask = ~(np.isnan(y) | X_all_df.isnull().any(axis=1).values)
                y = y[mask]
                X_cont = X_cont_df.values[mask]
                cont_names = continuous_preds.copy()

                if cat_dummies.shape[1] > 0:
                    X_cat = cat_dummies.values[mask]
                else:
                    X_cat = np.empty((len(y), 0))

                if len(y) < len(cont_names) + len(cat_dummy_names) + 2:
                    st.error(f"Not enough observations ({len(y)}) for "
                             f"{len(cont_names) + len(cat_dummy_names)} predictors.")
                    return

                # --- feature engineering (ONLY on continuous predictors) ----------
                if (include_squares or include_interactions) and len(cont_names) > 0:
                    X_cont_ext, cont_ext_names, cont_is_higher = generate_extended_features(
                        X_cont, cont_names,
                        include_squares=include_squares,
                        include_interactions=include_interactions,
                    )
                else:
                    X_cont_ext = X_cont.copy()
                    cont_ext_names = list(cont_names)
                    cont_is_higher = [False] * len(cont_names)

                # Append categorical dummies (no squares/interactions for dummies)
                if X_cat.shape[1] > 0:
                    X_ext = np.column_stack([X_cont_ext, X_cat])
                    ext_names = cont_ext_names + cat_dummy_names
                    is_higher = cont_is_higher + [False] * len(cat_dummy_names)
                else:
                    X_ext = X_cont_ext
                    ext_names = cont_ext_names
                    is_higher = cont_is_higher

                original_names = cont_names + cat_dummy_names

                # --- forward stepwise (always OLS) --------------------------------
                steps, selected_idx = forward_stepwise(
                    X_ext, y, ext_names, is_higher, alpha_enter=0.25,
                )

                # Use selected features for final model if OLS; else use all
                if model_key == "OLS" and selected_idx:
                    X_final = X_ext[:, selected_idx]
                    final_names = [ext_names[i] for i in selected_idx]
                elif selected_idx:
                    X_final = X_ext
                    final_names = ext_names
                    selected_idx = list(range(len(ext_names)))
                else:
                    # Stepwise found nothing — fall back to all linear terms
                    X_final = X_ext[:, :len(original_names)]
                    final_names = list(original_names)
                    selected_idx = list(range(len(original_names)))

                # --- fit model ---------------------------------------------------
                result = fit_model(X_final, y, final_names,
                                   model_type=model_key,
                                   alpha=reg_alpha,
                                   l1_ratio=l1_ratio,
                                   n_components=n_components)

                # --- diagnostics -------------------------------------------------
                impacts = compute_incremental_impact(
                    X_ext, y, ext_names, original_names, selected_idx,
                )
                if X_cont.shape[1] >= 2:
                    vif_r2 = compute_x_regressed_on_others(X_cont, cont_names)
                else:
                    vif_r2 = {n: 0.0 for n in cont_names}

                unusual = detect_unusual_data(result["residuals"], X_final)
                normality = check_residual_normality(result["residuals"])
                report_card = generate_report_card(
                    result["n"], result["p"], unusual, normality, response_var,
                )
                equation = format_equation(response_var,
                                           result["feature_names"],
                                           result["coefficients"])

                # Determine which original predictors are in the final model
                preds_in_model = []
                for orig in continuous_preds:
                    for fn in final_names:
                        if orig in fn:
                            preds_in_model.append(orig)
                            break

                # --- store everything --------------------------------------------
                st.session_state.mreg_res_result = result
                st.session_state.mreg_res_steps = steps
                st.session_state.mreg_res_selected = selected_idx
                st.session_state.mreg_res_impacts = impacts
                st.session_state.mreg_res_vif_r2 = vif_r2
                st.session_state.mreg_res_unusual = unusual
                st.session_state.mreg_res_normality = normality
                st.session_state.mreg_res_report_card = report_card
                st.session_state.mreg_res_equation = equation
                st.session_state.mreg_res_response = response_var
                st.session_state.mreg_res_continuous = continuous_preds
                st.session_state.mreg_res_original_names = original_names
                st.session_state.mreg_res_ext_names = ext_names
                st.session_state.mreg_res_final_names = final_names
                st.session_state.mreg_res_data = data.loc[mask].reset_index(drop=True)
                st.session_state.mreg_res_model_key = model_key
                st.session_state.mreg_res_preds_in_model = preds_in_model

                st.success("✅ Regression completed successfully!")

            except Exception as e:
                st.error(f"❌ Regression failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    # ==================================================================
    # RESULTS (3 report tabs)
    # ==================================================================
    if "mreg_res_result" not in st.session_state:
        return

    st.markdown("---")

    r = st.session_state.mreg_res_result
    resp = st.session_state.mreg_res_response
    orig_names = st.session_state.mreg_res_original_names
    cont_preds = st.session_state.mreg_res_continuous
    eq = st.session_state.mreg_res_equation
    m_data = st.session_state.mreg_res_data
    mk = st.session_state.mreg_res_model_key

    tab_summary, tab_build, tab_card, tab_diag = st.tabs([
        "📋 Summary Report",
        "🏗️ Model Building Report",
        "📝 Report Card",
        "🔍 Diagnostic Report",
    ])

    # ==================== SUMMARY REPORT ====================
    with tab_summary:
        # Title
        st.markdown(_html(f"""
        <div style="background:#1f3864;color:white;padding:10px 18px;border-radius:6px 6px 0 0;text-align:center;">
        <div style="font-size:16px;font-weight:bold;">Multiple Regression for {resp}</div>
        <div style="font-size:13px;">Summary Report</div>
        </div>
        """), unsafe_allow_html=True)

        # Row 1: p-value bar | Comments
        col_pval, col_comments = st.columns([1, 1])

        with col_pval:
            st.markdown(
                '<div style="font-weight:bold;font-size:13px;margin:10px 0 4px;">'
                'Is there a relationship between Y and the X variables?</div>',
                unsafe_allow_html=True,
            )
            ols = r["ols_stats"]
            fig_p = plot_pvalue_bar(ols["f_p_value"], alpha=0.10)
            st.plotly_chart(fig_p, use_container_width=True, key="mreg_pval")

            if ols["f_p_value"] < 0.10:
                st.markdown(
                    '<div style="font-size:12px;">The relationship between Y '
                    'and the X variables in the model is statistically '
                    'significant (p &lt; 0.10).</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="font-size:12px;">The relationship between Y '
                    'and the X variables is not statistically significant '
                    '(p &ge; 0.10).</div>',
                    unsafe_allow_html=True,
                )

        with col_comments:
            st.markdown(
                '<div style="font-weight:bold;font-size:13px;margin:10px 0 6px;">Comments</div>',
                unsafe_allow_html=True,
            )
            final_names = st.session_state.mreg_res_final_names
            terms_html = "".join(
                [f'<div style="margin-left:12px;">{n}</div>' for n in final_names]
            )
            st.markdown(
                f'<div style="border:1px solid #ccc;border-radius:4px;padding:12px;'
                f'background:#f8f9fa;font-size:12px;min-height:160px;">'
                f'<div style="margin-bottom:8px;">The following terms are in the fitted '
                f'equation that models the relationship between Y and the X variables:</div>'
                f'{terms_html}'
                f'<div style="margin-top:12px;">If the model fits the data well, this '
                f'equation can be used to predict <b>{resp}</b> for specific values of the '
                f'X variables, or find the settings for the X variables that correspond to '
                f'a desired value or range of values for <b>{resp}</b>.</div></div>',
                unsafe_allow_html=True,
            )

        # Row 2: R² gradient bar
        st.markdown(
            '<div style="font-weight:bold;font-size:13px;margin:16px 0 4px;">'
            '% of variation explained by the model</div>',
            unsafe_allow_html=True,
        )
        r_sq_pct = r["r_squared"] * 100.0
        _render_rsquared_bar(r_sq_pct)
        st.markdown(
            f'<div style="font-size:12px;margin-top:18px;">'
            f'{r_sq_pct:.2f}% of the variation in Y can be explained by the '
            f'regression model.</div>',
            unsafe_allow_html=True,
        )

        # Row 3: Scatter plots
        st.markdown(
            f'<div style="font-weight:bold;font-size:13px;margin:18px 0 6px;">'
            f'{resp} vs X Variables</div>',
            unsafe_allow_html=True,
        )
        preds_in = st.session_state.mreg_res_preds_in_model
        fig_scatter = plot_scatter_panels(m_data, resp, cont_preds, preds_in)
        st.plotly_chart(fig_scatter, use_container_width=True, key="mreg_scatter")
        st.caption("A gray background represents an X variable not in the model.")

    # ==================== MODEL BUILDING REPORT ====================
    with tab_build:
        # Title + subtitle
        subtitle_vars = " &nbsp;&nbsp; ".join(
            [f"X{i+1}: {n}" for i, n in enumerate(cont_preds)]
        )
        st.markdown(_html(f"""
        <div style="background:#1f3864;color:white;padding:10px 18px;border-radius:6px 6px 0 0;text-align:center;">
        <div style="font-size:16px;font-weight:bold;">Multiple Regression for {resp}</div>
        <div style="font-size:13px;">Model Building Report</div>
        </div>
        <div style="border:1px solid #ccc;border-top:none;padding:8px 14px;background:#f8f9fa;text-align:center;font-size:12px;">
        {subtitle_vars}
        </div>
        """), unsafe_allow_html=True)

        # Model type badge
        st.info(f"**Model type:** {mk}")

        # Equation
        st.markdown(
            f'<div style="font-weight:bold;font-size:13px;margin:12px 0 4px;">Final Model Equation</div>'
            f'<div style="font-family:monospace;font-size:12px;background:#f0f0f0;'
            f'padding:8px;border-radius:4px;overflow-x:auto;">{eq}</div>',
            unsafe_allow_html=True,
        )

        # Coefficient table
        with st.expander("📊 Coefficient Table", expanded=True):
            coef_df = pd.DataFrame({
                "Term": r["feature_names"],
                "Coefficient": r["coefficients"],
                "SE (OLS)": r["ols_stats"]["se"],
                "t-Value (OLS)": r["ols_stats"]["t_values"],
                "p-Value (OLS)": r["ols_stats"]["p_values"],
            })
            st.dataframe(
                coef_df.style.format({
                    "Coefficient": "{:.4f}",
                    "SE (OLS)": "{:.4f}",
                    "t-Value (OLS)": "{:.3f}",
                    "p-Value (OLS)": "{:.4f}",
                }),
                use_container_width=True,
            )
            ca, cb, cc = st.columns(3)
            with ca:
                st.metric("R²", f"{r['r_squared']*100:.2f}%")
            with cb:
                st.metric("R²(adj)", f"{r['r_squared_adj']*100:.2f}%")
            with cc:
                st.metric("S", f"{r['s']:.4f}")

        # Row: Model Building Sequence | Incremental + VIF charts
        col_build, col_charts = st.columns([1, 1])

        with col_build:
            steps = st.session_state.mreg_res_steps
            if steps:
                st.markdown(
                    '<div style="font-weight:bold;font-size:13px;margin-bottom:4px;">'
                    'Model Building Sequence</div>'
                    '<div style="font-size:11px;color:#666;margin-bottom:6px;">'
                    'Displays the order in which terms were added or removed.</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(_render_step_table(steps), unsafe_allow_html=True)
                fig_build = plot_model_building_sequence(steps)
                st.plotly_chart(fig_build, use_container_width=True,
                                key="mreg_build_chart")
            else:
                st.info("No stepwise model building steps recorded.")

        with col_charts:
            impacts = st.session_state.mreg_res_impacts
            st.markdown(
                '<div style="font-weight:bold;font-size:13px;margin-bottom:2px;">'
                'Incremental Impact of X Variables</div>'
                '<div style="font-size:11px;color:#666;margin-bottom:4px;">'
                'Long bars represent Xs that contribute the most new '
                'information to the model.</div>',
                unsafe_allow_html=True,
            )
            fig_impact = plot_incremental_impact(impacts)
            st.plotly_chart(fig_impact, use_container_width=True,
                            key="mreg_impact_chart")

            vif_r2 = st.session_state.mreg_res_vif_r2
            st.markdown(
                '<div style="font-weight:bold;font-size:13px;margin:12px 0 2px;">'
                'Each X Regressed on All Other Terms</div>'
                '<div style="font-size:11px;color:#666;margin-bottom:4px;">'
                'Long bars represent Xs that do not help explain additional '
                'variation in Y.</div>',
                unsafe_allow_html=True,
            )
            fig_vif = plot_x_regressed_on_others(vif_r2)
            st.plotly_chart(fig_vif, use_container_width=True,
                            key="mreg_vif_chart")

    # ==================== REPORT CARD ====================
    with tab_card:
        _render_report_card(st.session_state.mreg_res_report_card, resp)

        # Additional diagnostic expander
        with st.expander("📘 Interpretation Guide", expanded=False):
            st.markdown("""
**Amount of Data**
Larger samples provide more precise estimates of R² and coefficients. Minitab
recommends n ≥ 50 for stable estimates with multiple predictors.

**Unusual Data**
Large residuals (|standardized residual| > 2) indicate observations poorly
fit by the model. High-leverage points (unusual X values) can disproportionately
influence the fitted equation. Investigate and consider removing special-cause data.

**Normality**
With n ≥ 15 the Central Limit Theorem makes p-values reasonably robust even if
residuals are not perfectly normal. For smaller samples, check the Shapiro-Wilk
test and consider transforming the response.

**Multicollinearity (Each X Regressed on Others)**
If an X has high R² when regressed on all other Xs, it is nearly a linear
combination of the others (high VIF). This inflates coefficient standard errors
but does not affect prediction accuracy.

**Model Types**
- **OLS**: Standard ordinary least-squares. Best when predictors are uncorrelated.
- **Ridge**: Adds L2 penalty. Shrinks coefficients toward zero. Good for multicollinearity.
- **Lasso**: Adds L1 penalty. Can shrink some coefficients exactly to zero (feature selection).
- **ElasticNet**: Combines L1 + L2 penalties (controlled by L1 ratio).
- **PLS**: Projects X and Y into latent components. Excellent for wide data (p >> n), e.g. spectroscopy.
            """)

    # ==================== DIAGNOSTIC REPORT ====================
    with tab_diag:
        unusual = st.session_state.mreg_res_unusual

        st.markdown(_html(f"""
        <div style="background:#1f3864;color:white;padding:10px 18px;border-radius:6px 6px 0 0;text-align:center;">
        <div style="font-size:16px;font-weight:bold;">Multiple Regression for {resp}</div>
        <div style="font-size:13px;">Diagnostic Report</div>
        </div>
        """), unsafe_allow_html=True)

        # Legend
        st.markdown(
            '<div style="font-size:12px;margin:10px 0;">'
            '<span style="color:#e74c3c;font-weight:bold;">&#9679; Red</span> = Large residual '
            '&nbsp;&nbsp;&nbsp;'
            '<span style="color:#3498db;font-weight:bold;">&#9679; Blue</span> = High leverage '
            '&nbsp;&nbsp;&nbsp;'
            '<span style="color:#9b59b6;font-weight:bold;">&#9679; Purple</span> = Both'
            '</div>',
            unsafe_allow_html=True,
        )

        # Four-panel diagnostic chart
        fig_diag = plot_diagnostic_report(r, unusual, resp)
        st.plotly_chart(fig_diag, use_container_width=True, key="mreg_diag_chart")

        # Unusual observations table
        n_lr = unusual.get("n_large_residuals", 0)
        n_hl = unusual.get("n_high_leverage", 0)

        if n_lr > 0 or n_hl > 0:
            st.markdown(
                f'<div style="font-weight:bold;font-size:13px;margin:16px 0 6px;">'
                f'Unusual Observations ({n_lr} large residuals, {n_hl} high leverage)</div>',
                unsafe_allow_html=True,
            )

            # Build dataframe of flagged observations
            y_pred = r["y_pred"]
            residuals = r["residuals"]
            std_res = unusual.get("std_residuals", np.zeros(len(residuals)))
            leverage = unusual.get("leverage", np.zeros(len(residuals)))
            large_set = set(unusual.get("large_residuals", []))
            lever_set = set(unusual.get("high_leverage", []))

            flagged_rows = []
            for i in range(len(residuals)):
                if i in large_set or i in lever_set:
                    flags = []
                    if i in large_set:
                        flags.append("Large Residual")
                    if i in lever_set:
                        flags.append("High Leverage")
                    flagged_rows.append({
                        "Obs": i + 1,
                        "Fitted": round(float(y_pred[i]), 3),
                        "Residual": round(float(residuals[i]), 3),
                        "Std Residual": round(float(std_res[i]), 3),
                        "Leverage": round(float(leverage[i]), 4),
                        "Flag": "; ".join(flags),
                    })

            flag_df = pd.DataFrame(flagged_rows)

            def _colour_flag(row):
                flag = row["Flag"]
                if "Large Residual" in flag and "High Leverage" in flag:
                    return ["color: #9b59b6; font-weight: bold"] * len(row)
                elif "Large Residual" in flag:
                    return ["color: #e74c3c; font-weight: bold"] * len(row)
                elif "High Leverage" in flag:
                    return ["color: #3498db; font-weight: bold"] * len(row)
                return [""] * len(row)

            st.dataframe(
                flag_df.style.apply(_colour_flag, axis=1).format({
                    "Fitted": "{:.3f}",
                    "Residual": "{:.3f}",
                    "Std Residual": "{:.3f}",
                    "Leverage": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("✅ No unusual observations detected.")

    # ==================== EXCEL EXPORT ====================
    st.markdown("---")
    st.markdown("### 📥 Export All Reports to Excel")

    if st.button("💾 Generate Excel Report", type="primary",
                 use_container_width=True, key="mreg_export"):
        import tempfile, os
        try:
            tmp_path = os.path.join(tempfile.gettempdir(), f"mreg_{resp}_report.xlsx")
            export_results_to_excel(
                result=r,
                steps=st.session_state.mreg_res_steps,
                impacts=st.session_state.mreg_res_impacts,
                vif_r2=st.session_state.mreg_res_vif_r2,
                unusual=st.session_state.mreg_res_unusual,
                normality=st.session_state.mreg_res_normality,
                report_card=st.session_state.mreg_res_report_card,
                equation=eq,
                response_name=resp,
                filepath=tmp_path,
            )
            with open(tmp_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Excel Report",
                    data=f.read(),
                    file_name=f"Multiple_Regression_{resp}_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="mreg_download",
                )
            st.success("✅ Excel report generated! Click the download button above.")
        except Exception as e:
            st.error(f"❌ Export failed: {e}")
            import traceback
            st.code(traceback.format_exc())

    st.caption("© 2026 Roquette Advanced Analytics")
