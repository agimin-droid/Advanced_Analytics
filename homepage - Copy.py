"""
Roquette Advanced Analytics
Homepage - Main navigation and introduction
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Import all modules
try:
    import data_handling; DATA_HANDLING_AVAILABLE = True
except ImportError: DATA_HANDLING_AVAILABLE = False
try:
    import pca; PCA_AVAILABLE = True
except ImportError: PCA_AVAILABLE = False
try:
    import pca_monitoring_page; PCA_MONITORING_AVAILABLE = True
except ImportError: PCA_MONITORING_AVAILABLE = False
try:
    import mlr_doe; MLR_DOE_AVAILABLE = True
except ImportError: MLR_DOE_AVAILABLE = False
try:
    import univariate_page; UNIVARIATE_AVAILABLE = True
except ImportError: UNIVARIATE_AVAILABLE = False
try:
    import bivariate_page; BIVARIATE_AVAILABLE = True
except ImportError: BIVARIATE_AVAILABLE = False
try:
    import transformations; TRANSFORMATIONS_AVAILABLE = True
except ImportError: TRANSFORMATIONS_AVAILABLE = False
try:
    import eda_page; EDA_AVAILABLE = True
except ImportError: EDA_AVAILABLE = False
try:
    import ttest_page; TTEST_AVAILABLE = True
except ImportError: TTEST_AVAILABLE = False
try:
    import ml_page; ML_AVAILABLE = True
except ImportError: ML_AVAILABLE = False


def show_home():
    """Show the main homepage"""

    # ==================== PERFECT CENTERED HERO LOGO ====================
    logo_path = Path("assets/roquette_logo.png")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if logo_path.exists():
            st.image(str(logo_path), width=320)
        else:
            st.error("⚠️ Logo file not found in assets/roquette_logo.png")

    st.markdown("""
    <h1 style='text-align: center; font-size: 3.8rem; margin: 1rem 0 0.5rem 0; 
               background: linear-gradient(90deg, #003087, #00A3E0, #7CC142); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700;'>
        Roquette Advanced Analytics
    </h1>
    <p style='text-align: center; font-size: 1.35rem; color: #444; max-width: 900px; margin: 0 auto;'>
        Professional chemometric, statistical and machine learning tools for the life sciences industry
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.info("""
    ### Included Modules

    ✅ Data Handling & Import  
    ✅ PCA Analysis  
    ✅ Quality Control (PCA Monitoring)  
    ✅ MLR & DoE (Single Response)  
    ✅ Univariate Analysis  
    ✅ Bivariate Analysis  
    ✅ Preprocessing & Transformations  
    ✅ EDA - Exploratory Data Analysis  
    ✅ 2-Sample t-Test (Minitab Style)  
    ✅ 🤖 Machine Learning Analysis (Regression + Classification)

    For additional modules or enterprise deployment, please contact your Roquette Analytics team.
    """)

    st.markdown("---")

    if DATA_HANDLING_AVAILABLE:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📊 Start with Data Handling", use_container_width=True, key="cta_data_handling"):
                st.session_state.current_page = "Data Handling"
                st.rerun()

    st.markdown("## 🚀 Available Modules")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Core Analysis:**")
        if PCA_AVAILABLE and st.button("✅ 📈 PCA", use_container_width=True, key="btn_pca"):
            st.session_state.current_page = "PCA"; st.rerun()
        if PCA_MONITORING_AVAILABLE and st.button("✅ 📊 Quality Control", use_container_width=True, key="btn_qc"):
            st.session_state.current_page = "Quality Control"; st.rerun()
        if MLR_DOE_AVAILABLE and st.button("✅ 🧪 MLR & DoE", use_container_width=True, key="btn_mlr"):
            st.session_state.current_page = "MLR/DOE"; st.rerun()
        if ML_AVAILABLE and st.button("✅ 🤖 Machine Learning Analysis", use_container_width=True, key="btn_ml_home"):
            st.session_state.current_page = "ML Analysis"; st.rerun()

    with col2:
        st.markdown("**Statistical & Advanced:**")
        if UNIVARIATE_AVAILABLE and st.button("✅ 📉 Univariate", use_container_width=True, key="btn_uni"):
            st.session_state.current_page = "Univariate Analysis"; st.rerun()
        if BIVARIATE_AVAILABLE and st.button("✅ 🔗 Bivariate", use_container_width=True, key="btn_bi"):
            st.session_state.current_page = "Bivariate Analysis"; st.rerun()
        if TRANSFORMATIONS_AVAILABLE and st.button("✅ ⚙️ Preprocessing", use_container_width=True, key="btn_trans"):
            st.session_state.current_page = "Transformations"; st.rerun()
        if EDA_AVAILABLE and st.button("✅ 📊 EDA", use_container_width=True, key="btn_eda"):
            st.session_state.current_page = "EDA"; st.rerun()
        if TTEST_AVAILABLE and st.button("✅ ⚖️ 2-Sample t-Test", use_container_width=True, key="btn_ttest_home"):
            st.session_state.current_page = "2-Sample t-Test"; st.rerun()


def main_content():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"

    # ==================== SIDEBAR - PERFECT CENTERED LOGO + SPLIT SECTIONS ====================
    logo_path = Path("assets/roquette_logo.png")
    if logo_path.exists():
        st.sidebar.image(str(logo_path), width=160)
    else:
        st.sidebar.error("Logo not found")

    st.sidebar.markdown("---")

    # Navigation with same split as Home page
    st.sidebar.markdown("**Core Analysis**")
    if DATA_HANDLING_AVAILABLE and st.sidebar.button("📊 Data Handling", use_container_width=True, key="nav_data_handling"):
        st.session_state.current_page = "Data Handling"; st.rerun()
    if PCA_AVAILABLE and st.sidebar.button("📈 PCA", use_container_width=True, key="nav_pca"):
        st.session_state.current_page = "PCA"; st.rerun()
    if PCA_MONITORING_AVAILABLE and st.sidebar.button("📊 Quality Control", use_container_width=True, key="nav_qc"):
        st.session_state.current_page = "Quality Control"; st.rerun()
    if MLR_DOE_AVAILABLE and st.sidebar.button("🧪 MLR/DOE", use_container_width=True, key="nav_mlr_doe"):
        st.session_state.current_page = "MLR/DOE"; st.rerun()
    if ML_AVAILABLE and st.sidebar.button("🤖 ML Analysis", use_container_width=True, key="nav_ml"):
        st.session_state.current_page = "ML Analysis"; st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Statistical & Advanced**")
    if UNIVARIATE_AVAILABLE and st.sidebar.button("📉 Univariate", use_container_width=True, key="nav_univariate"):
        st.session_state.current_page = "Univariate Analysis"; st.rerun()
    if BIVARIATE_AVAILABLE and st.sidebar.button("🔗 Bivariate", use_container_width=True, key="nav_bivariate"):
        st.session_state.current_page = "Bivariate Analysis"; st.rerun()
    if TRANSFORMATIONS_AVAILABLE and st.sidebar.button("⚙️ Preprocessing", use_container_width=True, key="nav_transformations"):
        st.session_state.current_page = "Transformations"; st.rerun()
    if EDA_AVAILABLE and st.sidebar.button("📊 EDA", use_container_width=True, key="nav_eda"):
        st.session_state.current_page = "EDA"; st.rerun()
    if TTEST_AVAILABLE and st.sidebar.button("⚖️ 2-Sample t-Test", use_container_width=True, key="nav_ttest"):
        st.session_state.current_page = "2-Sample t-Test"; st.rerun()

    st.sidebar.markdown("---")

    # Dataset selector
    with st.sidebar:
        st.markdown("### 📂 Current Dataset")
        from workspace_utils import get_workspace_datasets, activate_dataset_in_workspace
        available_datasets = get_workspace_datasets()
        if available_datasets:
            dataset_names = list(available_datasets.keys())
            current = st.session_state.get('dataset_name', None)
            default_idx = dataset_names.index(current) if current in dataset_names else 0
            selected = st.selectbox("🔄 Switch Dataset", dataset_names, index=default_idx, key="sidebar_dataset_selector")
            if selected:
                activate_dataset_in_workspace(selected, available_datasets[selected])
                st.markdown(f"**Name:** `{selected}`")
                st.markdown(f"**Samples:** {len(available_datasets[selected])}")
                st.markdown(f"**Variables:** {len(available_datasets[selected].columns)}")
        else:
            st.info("📊 Load a dataset in Data Handling")

    st.sidebar.markdown("---")
    st.sidebar.caption("© 2026 Roquette Advanced Analytics")

    # Routing
    if st.session_state.current_page == "Home":
        show_home()
    elif st.session_state.current_page == "Data Handling" and DATA_HANDLING_AVAILABLE:
        data_handling.show()
    elif st.session_state.current_page == "PCA" and PCA_AVAILABLE:
        pca.show()
    elif st.session_state.current_page == "Quality Control" and PCA_MONITORING_AVAILABLE:
        pca_monitoring_page.show()
    elif st.session_state.current_page == "MLR/DOE" and MLR_DOE_AVAILABLE:
        mlr_doe.show()
    elif st.session_state.current_page == "Univariate Analysis" and UNIVARIATE_AVAILABLE:
        univariate_page.show()
    elif st.session_state.current_page == "Bivariate Analysis" and BIVARIATE_AVAILABLE:
        bivariate_page.show()
    elif st.session_state.current_page == "Transformations" and TRANSFORMATIONS_AVAILABLE:
        transformations.show()
    elif st.session_state.current_page == "EDA" and EDA_AVAILABLE:
        eda_page.show()
    elif st.session_state.current_page == "2-Sample t-Test" and TTEST_AVAILABLE:
        ttest_page.show()
    elif st.session_state.current_page == "ML Analysis" and ML_AVAILABLE:
        ml_page.show()
    else:
        st.error(f"Page '{st.session_state.current_page}' not found")
        st.session_state.current_page = "Home"
        st.rerun()


def main():
    st.set_page_config(
        page_title="Roquette Advanced Analytics",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    main_content()


if __name__ == "__main__":
    main()