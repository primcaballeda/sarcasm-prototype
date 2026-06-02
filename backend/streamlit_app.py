"""
Sarcasm Detection System - Streamlit App
Main entry point for the application.

This is a clean, modular entry point that orchestrates the UI tabs.
All business logic is in the core, ui, and utils modules.
"""

import streamlit as st

import app as backend_app
from config import APP_TITLE, APP_ICON, APP_LAYOUT
from ui.styles import apply_custom_style
from ui.tabs import render_predict_tab, render_batch_testing_tab, render_analytics_tab
from utils.helpers import init_session_state


# Page configuration
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=APP_LAYOUT)


def main() -> None:
    """Main entry point for the Streamlit app."""
    # Initialize styling and session state
    apply_custom_style()
    init_session_state()

    # Render header
    st.markdown(
        """
        <div class='title-wrap'>
            <h1>Sarcasm Detection System</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        "<p class='subtitle'>GloVe + CNN + BiLSTM + Attention vs. BERT + CNN + BiLSTM + Multi-Head Attention</p>",
        unsafe_allow_html=True
    )

    # Show model load status
    status_fn = getattr(backend_app, "get_model_status", None)
    if callable(status_fn):
        status = status_fn()
        proposed_status = (status.get("models") or {}).get("proposed") or {}
        baseline_status = (status.get("models") or {}).get("baseline") or {}
        if not baseline_status.get("loaded") and baseline_status.get("error"):
            st.error(f"GloVe Model failed to load: {baseline_status.get('error')}")
        if not proposed_status.get("loaded") and proposed_status.get("error"):
            st.warning(f"BERT Model not loaded: {proposed_status.get('error')}")

    # Create tabs
    tab_predict, tab_batch, tab_analytics = st.tabs([
        "**PREDICT**",
        "**BATCH TESTING**",
        "**MODEL ANALYTICS**"
    ])

    # Render tabs
    with tab_predict:
        render_predict_tab()

    with tab_batch:
        render_batch_testing_tab()

    with tab_analytics:
        render_analytics_tab()


if __name__ == "__main__":
    main()
