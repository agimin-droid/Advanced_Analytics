"""
Machine Learning Analysis Page – Multi-Model PyCaret Style
With Data Quality Check and 7 classification models
"""

import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix   # ← FIXED: this line was missing

try:
    from ml_utils.ml_analysis import (
        train_multiple_models,
        plot_actual_vs_predicted,
        plot_residuals,
        plot_confusion_matrix,
        plot_feature_importance
    )
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    st.error(f"❌ ml_utils import failed: {e}")

try:
    from workspace_utils import get_workspace_datasets
    WORKSPACE_AVAILABLE = True
except ImportError:
    WORKSPACE_AVAILABLE = False


def show():
    if not ML_AVAILABLE:
        st.error("❌ Machine Learning module is not available.")
        return

    st.markdown("# 🤖 Machine Learning Analysis")
    st.markdown("**Multi-model training with PyCaret-style comparison**")

    st.markdown("---")

    if WORKSPACE_AVAILABLE:
        available_datasets = get_workspace_datasets()
        if not available_datasets:
            st.warning("⚠️ No datasets available. Load data in **Data Handling** first.")
            return
        dataset_name = st.selectbox("Select dataset:", list(available_datasets.keys()), key="ml_dataset")
        data = available_datasets[dataset_name]
    else:
        if 'current_data' not in st.session_state or st.session_state.current_data is None:
            st.error("❌ No data available")
            return
        data = st.session_state.current_data
        dataset_name = "Current Data"

    st.success(f"✅ Using: **{dataset_name}** ({len(data)} samples × {len(data.columns)} variables)")

    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    cat_cols = [col for col in data.columns if col not in numeric_cols or data[col].nunique() < 20]

    task = st.radio("Task Type:", ["Regression", "Classification"], horizontal=True, key="ml_task")

    target_options = numeric_cols if task == "Regression" else cat_cols
    target_col = st.selectbox("Select Target Variable:", target_options, key="ml_target")

    feature_cols = [col for col in data.columns if col != target_col]
    selected_features = st.multiselect("Features (default = all others):", feature_cols, default=feature_cols)

    test_size = st.slider("Test set size (%)", 10, 50, 25, key="ml_testsize") / 100.0

    if task == "Regression":
        available_models = ["Linear Regression", "Ridge Regression", "Random Forest", "Gradient Boosting"]
    else:
        available_models = [
            "Logistic Regression", "Decision Tree", "K-Nearest Neighbors",
            "Random Forest", "SVM", "Gradient Boosting", "AdaBoost"
        ]

    selected_models = st.multiselect(
        "Select models to train:",
        available_models,
        default=available_models[:3]
    )

    if st.button("🚀 Train All Selected Models", type="primary", use_container_width=True):
        if not selected_models:
            st.error("Please select at least one model.")
        else:
            df_ml = data[selected_features + [target_col]].copy()

            # Data Quality Check
            st.subheader("🔍 Dataset Quality Check (before training)")
            nan_features = df_ml.drop(columns=[target_col]).isna().sum().sum()
            nan_target = df_ml[target_col].isna().sum()
            if nan_features + nan_target == 0:
                st.success("✅ No missing values detected. Ready for training.")
            else:
                st.warning(f"⚠️ Found **{nan_features}** missing values in features and **{nan_target}** in target.")
                st.info("**Automatic fix applied:** Numeric → median, Categorical → most frequent value.")

            # Training
            with st.spinner("Training models..."):
                result = train_multiple_models(
                    df_ml, target_col, selected_models, task, test_size=test_size
                )

            st.success("🎉 All models trained successfully!")

            st.subheader("🏆 Model Comparison")
            st.dataframe(
                result['comparison_df'].style.highlight_max(subset=result['comparison_df'].columns[1:], axis=0, color='#90EE90'),
                use_container_width=True,
                hide_index=True
            )

            best_model_name = result['comparison_df'].iloc[0]['Model']
            best_result = result['detailed_results'][best_model_name]

            st.subheader(f"🔥 Best Model: **{best_model_name}**")

            col1, col2 = st.columns(2)
            with col1:
                if task == "Regression":
                    st.plotly_chart(plot_actual_vs_predicted(best_result['y_test'], best_result['y_pred']), use_container_width=True)
                    st.plotly_chart(plot_residuals(best_result['y_test'], best_result['y_pred']), use_container_width=True)
                else:
                    cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
                    class_names = result['label_encoder'].classes_ if result.get('label_encoder') is not None else None
                    st.plotly_chart(plot_confusion_matrix(cm, class_names), use_container_width=True)

            with col2:
                if best_result.get('feature_importance') is not None:
                    st.plotly_chart(plot_feature_importance(best_result['feature_importance']), use_container_width=True)

            st.download_button(
                "📥 Download Full Comparison Table (CSV)",
                data=result['comparison_df'].to_csv(index=False),
                file_name="ml_model_comparison.csv",
                mime="text/csv"
            )

    st.markdown("---")
    st.caption("© 2026 Roquette - Advanced Analytics • 7 classification models with automatic NaN handling")