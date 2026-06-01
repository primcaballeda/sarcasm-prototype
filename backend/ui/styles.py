"""Streamlit page styling and CSS."""

import streamlit as st


def apply_custom_style() -> None:
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

            /* Animations */
            @keyframes slideInDown {
                from {
                    opacity: 0;
                    transform: translateY(-20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            @keyframes slideInUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }

            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }

            @keyframes shimmer {
                0% { background-position: -1000px 0; }
                100% { background-position: 1000px 0; }
            }

            @keyframes popIn {
                0% {
                    opacity: 0;
                    transform: scale(0.95);
                }
                50% {
                    transform: scale(1.02);
                }
                100% {
                    opacity: 1;
                    transform: scale(1);
                }
            }

            @keyframes glow {
                0%, 100% { box-shadow: 0 6px 20px rgba(124, 58, 237, 0.35); }
                50% { box-shadow: 0 8px 32px rgba(124, 58, 237, 0.5); }
            }

            /* Modern theme - Pink, Purple, Blue with solid colors */
            .stApp {
                background: #f8f5fb;
                font-family: 'Inter', sans-serif;
            }

            .block-container {
                padding-top: 2.5rem;
                padding-bottom: 2.5rem;
                max-width: 1400px;
                animation: fadeIn 0.6s ease-out;
            }

            /* ========== HEADER SECTION ========== */
            .title-wrap {
                text-align: center;
                margin-bottom: 2rem;
                padding: 50px 30px;
                background: #ffffff;
                border-radius: 20px;
                border: 2px solid #7c3aed;
                box-shadow: 0 8px 32px rgba(124, 58, 237, 0.12);
                backdrop-filter: blur(20px);
                animation: slideInDown 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
            }

            .title-wrap h1 {
                margin: 0;
                font-weight: 800;
                letter-spacing: -0.03em;
                font-size: 3.5rem;
                color: #7c3aed;
                animation: slideInDown 1s cubic-bezier(0.34, 1.56, 0.64, 1) 0.1s backwards;
            }

            .subtitle {
                color: #7c3aed;
                text-align: center;
                margin-top: 15px;
                margin-bottom: 0;
                font-size: 1rem;
                font-weight: 600;
                letter-spacing: 0.5px;
                animation: slideInDown 1s cubic-bezier(0.34, 1.56, 0.64, 1) 0.2s backwards;
            }

            /* ========== CONTAINERS & CARDS ========== */
            div[data-testid="stVerticalBlockBorderWrapper"] {
                border: 2px solid #e0d9f0;
                border-radius: 16px;
                background: #ffffff;
                box-shadow: 0 4px 16px rgba(124, 58, 237, 0.08);
                transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
                backdrop-filter: blur(20px);
                animation: slideInUp 0.6s ease-out;
            }

            div[data-testid="stVerticalBlockBorderWrapper"]:hover {
                border-color: #7c3aed;
                box-shadow: 0 12px 40px rgba(124, 58, 237, 0.2);
                background: #faf8ff;
                transform: translateY(-4px);
            }

            /* ========== BUTTONS ========== */
            div.stButton > button {
                background: #7c3aed !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 10px !important;
                font-weight: 700 !important;
                letter-spacing: 0.5px;
                padding: 14px 32px !important;
                font-size: 15px !important;
                box-shadow: 0 6px 20px rgba(124, 58, 237, 0.35) !important;
                transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
                cursor: pointer !important;
                text-transform: uppercase;
                position: relative;
                overflow: hidden;
                animation: popIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
                text-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important;
            }

            div.stButton > button p {
                color: #ffffff !important;
                font-weight: 700 !important;
            }

            div.stButton > button::before {
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                width: 0;
                height: 0;
                background: rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                transform: translate(-50%, -50%);
                transition: width 0.6s, height 0.6s;
            }

            div.stButton > button:hover::before {
                width: 300px;
                height: 300px;
            }

            div.stButton > button:hover {
                background: #8b5cf6 !important;
                box-shadow: 0 8px 32px rgba(124, 58, 237, 0.5) !important;
                transform: translateY(-3px) scale(1.02) !important;
            }

            div.stButton > button:active {
                transform: translateY(-1px) scale(0.98) !important;
                box-shadow: 0 4px 16px rgba(124, 58, 237, 0.35) !important;
            }

            /* ========== TEXT INPUTS ========== */
            div[data-testid="stTextArea"] textarea {
                background: #ffffff !important;
                border: 2px solid #e0d9f0 !important;
                border-radius: 10px !important;
                color: #2d2d3d !important;
                font-family: 'Inter', sans-serif !important;
                font-size: 14px !important;
                padding: 16px !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
            }

            div[data-testid="stTextArea"] textarea::placeholder {
                color: #b8a8d4 !important;
            }

            div[data-testid="stTextArea"] textarea:focus {
                border-color: #7c3aed !important;
                box-shadow: 0 0 0 4px rgba(124, 58, 237, 0.15), 0 2px 8px rgba(0, 0, 0, 0.04) !important;
                background: #faf8ff !important;
            }

            /* ========== PROGRESS BARS ========== */
            div[data-testid="stProgress"] div[role="progressbar"] > div {
                background: #7c3aed !important;
                border-radius: 10px !important;
                height: 10px !important;
                box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);
                animation: shimmer 2s infinite;
                background-size: 1000px 100%;
            }

            /* ========== METRICS ========== */
            div[data-testid="stMetric"] {
                background: #ffffff;
                border: 2px solid #e0d9f0;
                border-radius: 12px;
                padding: 28px;
                box-shadow: 0 4px 16px rgba(124, 58, 237, 0.08);
                backdrop-filter: blur(20px);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                animation: slideInUp 0.6s ease-out;
            }

            div[data-testid="stMetric"]:hover {
                box-shadow: 0 8px 32px rgba(124, 58, 237, 0.15);
                transform: translateY(-2px);
                border-color: #7c3aed;
            }

            /* ========== RESULT BADGES ========== */
            .status-ok {
                color: #059669;
                font-weight: 700;
                background: #f0fdf4;
                padding: 14px 20px;
                border-radius: 10px;
                border: 2px solid #22c55e;
                display: inline-block;
                margin: 8px 0;
                font-size: 13px;
                letter-spacing: 0.4px;
                box-shadow: 0 4px 12px rgba(34, 197, 94, 0.15);
                animation: popIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
            }

            .status-bad {
                color: #dc2626;
                font-weight: 700;
                background: #fef2f2;
                padding: 14px 20px;
                border-radius: 10px;
                border: 2px solid #ef4444;
                display: inline-block;
                margin: 8px 0;
                font-size: 13px;
                letter-spacing: 0.4px;
                box-shadow: 0 4px 12px rgba(220, 38, 38, 0.15);
                animation: popIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
            }

            /* ========== TABS ========== */
            div[role="tablist"] {
                background: #ffffff;
                border-bottom: 2px solid #e0d9f0;
                padding-bottom: 0;
                border-radius: 12px 12px 0 0;
            }

            div[role="tab"] {
                color: #8b7a9b !important;
                border-radius: 8px 8px 0 0;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                font-weight: 600;
                font-size: 15px;
                padding: 14px 24px !important;
            }

            div[role="tab"][aria-selected="true"] {
                color: #7c3aed !important;
                border-bottom: 3px solid #7c3aed !important;
                background: #faf8ff;
                animation: popIn 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
            }

            div[role="tab"]:hover {
                color: #7c3aed !important;
                background: #f5f0ff;
            }

            /* ========== EXPANDERS ========== */
            div[data-testid="stExpander"] {
                border: 2px solid #e0d9f0;
                border-radius: 12px;
                background: #ffffff;
                box-shadow: 0 4px 16px rgba(124, 58, 237, 0.08);
                transition: all 0.3s ease;
            }

            div[data-testid="stExpander"]:hover {
                border-color: #7c3aed;
                box-shadow: 0 8px 24px rgba(124, 58, 237, 0.12);
            }

            /* ========== CHARTS & TABLES ========== */
            div[data-testid="stDataFrame"] {
                background: #ffffff !important;
                border-radius: 12px !important;
                border: 2px solid #e0d9f0 !important;
                box-shadow: 0 4px 16px rgba(124, 58, 237, 0.08) !important;
            }

            /* ========== ALERTS ========== */
            div[data-testid="stAlert"] {
                border-radius: 12px;
                border: 2px solid #e0d9f0;
                background: #ffffff !important;
                box-shadow: 0 4px 16px rgba(124, 58, 237, 0.08);
            }

            /* ========== DIVIDER ========== */
            hr {
                border: 0;
                height: 1.5px;
                background: #e0d9f0;
                margin: 36px 0 !important;
            }

            /* ========== TEXT STYLING ========== */
            h1, h2, h3, h4, h5, h6 {
                color: #2d2d3d !important;
                font-weight: 700 !important;
                letter-spacing: -0.02em;
                animation: slideInUp 0.6s ease-out;
            }

            p, li, span, label {
                color: #4a4a5e !important;
                line-height: 1.6;
            }

            .small-muted {
                color: #8b7a9b !important;
                font-size: 0.85rem !important;
                font-weight: 500 !important;
            }

            /* ========== FILE UPLOADER ========== */
            div[data-testid="stFileUploadDropzone"] {
                background: #ffffff !important;
                border: 2px dashed #7c3aed !important;
                border-radius: 12px !important;
                transition: all 0.3s ease;
            }

            div[data-testid="stFileUploadDropzone"]:hover {
                border-color: #d946a6 !important;
                background: #faf8ff !important;
                box-shadow: 0 4px 16px rgba(124, 58, 237, 0.12);
            }

            /* ========== SELECTBOX & MULTISELECT ========== */
            div[data-testid="stSelectbox"] > div > div {
                background: #ffffff !important;
                border: 2px solid #e0d9f0 !important;
                border-radius: 10px !important;
                color: #2d2d3d !important;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
                transition: all 0.2s ease;
            }

            div[data-testid="stSelectbox"] > div > div:focus {
                border-color: #7c3aed !important;
                box-shadow: 0 0 0 4px rgba(124, 58, 237, 0.15), 0 2px 8px rgba(0, 0, 0, 0.04) !important;
            }

            /* ========== CHECKBOX ========== */
            div[data-testid="stCheckbox"] label {
                color: #4a4a5e !important;
                font-weight: 500;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
