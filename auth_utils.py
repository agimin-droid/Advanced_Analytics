"""
Authentication utilities for ChemometricSolutions application.
Password protection has been COMPLETELY REMOVED for local/personal use.
MIT License © 2026 – You now have full unrestricted access on your PC.
"""

import streamlit as st

# ============================================================================
# CONFIGURATION - AUTHENTICATION DISABLED
# ============================================================================

# No password, no lockout, no form – direct access to all modules

# ============================================================================
# SESSION STATE SETUP
# ============================================================================

def initialize_session_state():
    """Force authenticated = True immediately for seamless local use."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = True
    if 'login_attempts' not in st.session_state:
        st.session_state.login_attempts = 0
    if 'lockout_time' not in st.session_state:
        st.session_state.lockout_time = None

# ============================================================================
# BYPASSED AUTHENTICATION FUNCTIONS
# ============================================================================

def is_locked_out():
    """Never locked out – authentication is disabled."""
    return False, 0

def verify_password(password):
    """Placeholder – always returns True (never called)."""
    return True

def handle_login_attempt(password):
    """Always succeeds (kept only for compatibility)."""
    st.session_state.authenticated = True
    return True

def logout():
    """Optional reset – not needed for local use."""
    st.session_state.authenticated = False
    st.session_state.login_attempts = 0
    st.session_state.lockout_time = None

def render_login_page():
    """
    CRITICAL FUNCTION CALLED BY streamlit_app.py
    Forces authentication and returns True immediately.
    No login screen is ever shown.
    """
    initialize_session_state()
    st.session_state.authenticated = True
    return True

def show_logout_button():
    """No logout button needed – you stay logged in forever locally."""
    pass