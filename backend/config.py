"""Global configuration and constants for the Sarcasm Detection app."""

import os

# App Configuration
APP_TITLE = "Sarcasm Detection System"
APP_ICON = "SD"
APP_LAYOUT = "wide"

# Model Names
BASELINE_MODEL_NAME = "GloVe + CNN + BiLSTM + Attention"
PROPOSED_MODEL_NAME = "BERT + CNN + BiLSTM + Multi-Head Attention"

# Processing Limits
MAX_WORDS = 200
MAX_DATASET_SAMPLES = 200
BATCH_PROCESSING_SIZE = 10

# File Paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "model")
BASELINE_METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")
PROPOSED_METRICS_PATH = os.path.join(MODEL_DIR, "proposed_model_metrics.json")

# Example texts for demo
EXAMPLES = [
    "Oh great, another Monday morning meeting!",
    "Yeah right, like that's ever going to happen...",
    "Yeah, I absolutely love working on weekends while everyone else is out relaxing. ",
    "Thank you for your help today. I appreciate it.",
]

# Color Palette
COLOR_PRIMARY = "#374151"
COLOR_SUCCESS = "#22c55e"
COLOR_ERROR = "#ef4444"
COLOR_WARNING = "#f59e0b"
COLOR_PURPLE = "#374151"
COLOR_CYAN = "#6b7280"

# Confusion Matrix Baseline (for reference/analytics)
CONFUSION_MATRIX_BASELINE = {
    "truePositive": 651,
    "falsePositive": 280,
    "falseNegative": 277,
    "trueNegative": 670,
}

# Confusion Matrix Proposed (for reference/analytics)
CONFUSION_MATRIX_PROPOSED = {
    "truePositive": 681,
    "falsePositive": 196,
    "falseNegative": 258,
    "trueNegative": 743,
}

# CSV Expected Headers
EXPECTED_CSV_HEADERS = ["corpus", "label", "id", "response text"]
EXPECTED_LABEL_VALUES = {"sarc", "notsarc"}

# Initial session state defaults
SESSION_STATE_DEFAULTS = {
    "text": "",
    "results": None,
    "dataset": [],
    "dataset_results": [],
    "show_all_results": False,
    "upload_status": {
        "type": "neutral",
        "message": "No dataset uploaded yet.",
        "fileName": "",
    },
    "uploaded_signature": None,
    "current_example": None,
}
