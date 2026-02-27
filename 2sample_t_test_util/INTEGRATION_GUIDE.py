# ============================================================
# INTEGRATION GUIDE: Adding Two-Sample T-Test to homepage.py
# ============================================================
#
# This file shows the EXACT lines to add/modify in homepage.py.
# There are 4 insertion points.
#
# ============================================================

# ---- CHANGE 1: Add import (after the univariate_page import block, ~line 44) ----
# Insert AFTER:
#   try:
#       import univariate_page
#       UNIVARIATE_AVAILABLE = True
#   except ImportError:
#       UNIVARIATE_AVAILABLE = False
#
# ADD:

try:
    import two_sample_ttest_page
    TTEST_AVAILABLE = True
except ImportError:
    TTEST_AVAILABLE = False


# ---- CHANGE 2: Add button on homepage show_home() (inside col2, ~line 191) ----
# Insert AFTER the Univariate button block:
#   if UNIVARIATE_AVAILABLE and st.button("✅ 📉 Univariate ...", ...):
#       ...
#
# ADD:

        # Two-Sample T-Test - AVAILABLE
        if TTEST_AVAILABLE and st.button("✅ 🔬 Two-Sample T-Test", use_container_width=True, key="btn_ttest"):
            st.session_state.current_page = "Two-Sample T-Test"
            st.rerun()


# ---- CHANGE 3: Add sidebar nav button (after Univariate nav, ~line 379) ----
# Insert AFTER:
#   if UNIVARIATE_AVAILABLE:
#       if st.sidebar.button("📉 Univariate", ...):
#           ...
#
# ADD:

    if TTEST_AVAILABLE:
        if st.sidebar.button("🔬 T-Test", use_container_width=True, key="nav_ttest"):
            st.session_state.current_page = "Two-Sample T-Test"
            st.rerun()


# ---- CHANGE 4: Add routing (in main_content(), ~line 494) ----
# Insert AFTER:
#   elif st.session_state.current_page == "Univariate Analysis" and UNIVARIATE_AVAILABLE:
#       univariate_page.show()
#
# ADD:

    elif st.session_state.current_page == "Two-Sample T-Test" and TTEST_AVAILABLE:
        two_sample_ttest_page.show()


# ---- CHANGE 5 (optional): Update the info banner in show_home() (~line 105) ----
# Add to the list of included modules:
#   ✅ Two-Sample T-Test (Minitab-equivalent)
