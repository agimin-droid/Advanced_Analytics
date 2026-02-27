"""
Machine Learning Core – Multi-model training with PyCaret-style comparison
Automatic NaN handling + 7 classification models
"""

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==================== MODELS ====================
REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42)
}

CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "SVM": SVC(probability=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42)
}


def train_multiple_models(data: pd.DataFrame, target_col: str, model_names: list, task: str,
                          test_size: float = 0.25, random_state: int = 42):
    """Train multiple models with automatic NaN handling."""

    # ==================== AUTOMATIC NaN HANDLING ====================
    X = data.drop(columns=[target_col]).copy()
    y = data[target_col].copy()

    numeric_cols = X.select_dtypes(include=['number']).columns
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns

    for col in numeric_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    for col in cat_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].mode()[0])
    if y.isna().any():
        if task == "Regression":
            y = y.fillna(y.median())
        else:
            y = y.fillna(y.mode()[0])

    X = pd.get_dummies(X, drop_first=True)

    # ==================== SPLIT ====================
    if task == "Classification":
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    else:
        le = None
        y_encoded = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state,
        stratify=y_encoded if task == "Classification" else None
    )

    # ==================== TRAINING WITH PROGRESS ====================
    results = {}
    comparison_rows = []
    elapsed_times = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, name in enumerate(model_names):
        model_start = time.time()
        status_text.text(f"Training {name}... ({i+1}/{len(model_names)})")

        if task == "Regression":
            model = REGRESSION_MODELS[name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = {
                "R²": r2_score(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
            }
        else:
            model = CLASSIFICATION_MODELS[name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1-Score": f1_score(y_test, y_pred, average='weighted')
            }

        elapsed = time.time() - model_start
        elapsed_times.append(elapsed)

        progress = (i + 1) / len(model_names)
        progress_bar.progress(progress)

        avg_time = np.mean(elapsed_times)
        remaining = avg_time * (len(model_names) - i - 1)
        status_text.text(f"✅ {name} done in {elapsed:.1f}s | Est. remaining: {remaining:.1f}s")

        importance = None
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

        results[name] = {
            'model': model,
            'metrics': metrics,
            'y_test': y_test,
            'y_pred': y_pred,
            'feature_importance': importance
        }

        row = {'Model': name}
        row.update(metrics)
        comparison_rows.append(row)

    progress_bar.progress(1.0)
    status_text.text("🎉 All models trained successfully!")

    comparison_df = pd.DataFrame(comparison_rows)
    sort_col = "R²" if task == "Regression" else "Accuracy"
    comparison_df = comparison_df.sort_values(sort_col, ascending=False)

    return {
        'comparison_df': comparison_df,
        'detailed_results': results,
        'label_encoder': le
    }


# Plotting functions
def plot_actual_vs_predicted(y_test, y_pred, title="Actual vs Predicted"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', marker=dict(size=8, opacity=0.7)))
    minv = min(y_test.min(), y_pred.min())
    maxv = max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(x=[minv, maxv], y=[minv, maxv], mode='lines', line=dict(dash='dash')))
    fig.update_layout(title=title, xaxis_title="Actual", yaxis_title="Predicted", template="plotly_white", height=480)
    return fig


def plot_residuals(y_test, y_pred):
    residuals = np.array(y_test) - np.array(y_pred)
    fig = px.histogram(residuals, nbins=30, title="Residual Distribution")
    fig.update_layout(template="plotly_white", xaxis_title="Residual", height=400)
    return fig


def plot_confusion_matrix(cm, class_names):
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                    labels=dict(x="Predicted", y="Actual"))
    fig.update_xaxes(ticktext=class_names, tickvals=list(range(len(class_names))))
    fig.update_yaxes(ticktext=class_names, tickvals=list(range(len(class_names))))
    fig.update_layout(title="Confusion Matrix", template="plotly_white", height=480)
    return fig


def plot_feature_importance(importance_df, top_n=15):
    top = importance_df.head(top_n)
    fig = px.bar(top, x='Importance', y='Feature', orientation='h', title=f"Top {top_n} Feature Importance")
    fig.update_layout(template="plotly_white", height=480, yaxis={'categoryorder': 'total ascending'})
    return fig