"""Model prediction functions."""

from typing import Any, Dict

import app as backend_app
from config import BASELINE_MODEL_NAME, PROPOSED_MODEL_NAME
from core.validation import validate_input


def format_model_result(raw: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Format raw model output into standardized result format.
    
    Args:
        raw: Raw model output
        model_name: Either 'baseline' or 'proposed'
        
    Returns:
        Formatted result dictionary
    """
    raw_error = raw.get("error") if isinstance(raw, dict) else None
    
    if model_name == "baseline":
        if raw_error:
            return {
                "isSarcastic": False,
                "confidence": 0.0,
                "indicators": [
                    "GloVe model is unavailable in this deployment.",
                    f"Details: {raw_error}",
                ],
                "model": BASELINE_MODEL_NAME,
                "processingTime": raw.get("processingTime", "N/A") if isinstance(raw, dict) else "N/A",
                "error": raw_error,
            }
        return {
            "isSarcastic": bool(raw.get("isSarcastic", False)),
            "confidence": float(raw.get("confidence", 0.0)),
            "indicators": [
                "Real Keras model loaded",
                "GloVe embeddings processed",
                "BiLSTM with attention mechanism",
                f"Sarcasm probability: {raw.get('probabilities', {}).get('sarcastic', 0)}%",
                f"Non-sarcasm probability: {raw.get('probabilities', {}).get('not_sarcastic', 0)}%",
            ],
            "model": BASELINE_MODEL_NAME,
            "processingTime": raw.get("processingTime", "N/A"),
        }

    # Proposed model
    if raw_error:
        return {
            "isSarcastic": False,
            "confidence": 0.0,
            "indicators": [
                "BERT model is unavailable in this deployment.",
                f"Details: {raw_error}",
                "If deployed on Streamlit Cloud, the .pt weights may be missing (often ignored by git).",
                "Set env var SARCASM_PROPOSED_MODEL_URL to a direct-download URL for sarcasm_model.pt.",
            ],
            "model": PROPOSED_MODEL_NAME,
            "processingTime": raw.get("processingTime", "N/A") if isinstance(raw, dict) else "N/A",
            "error": raw_error,
        }

    return {
        "isSarcastic": bool(raw.get("isSarcastic", False)),
        "confidence": float(raw.get("confidence", 0.0)),
        "indicators": [
            "Real PyTorch model loaded",
            "BERT contextual embeddings analyzed",
            "CNN + BiLSTM architecture",
            "Multi-head attention patterns detected",
            f"Sarcasm probability: {raw.get('probabilities', {}).get('sarcastic', 0)}%",
            f"Non-sarcasm probability: {raw.get('probabilities', {}).get('not_sarcastic', 0)}%",
        ],
        "model": PROPOSED_MODEL_NAME,
        "processingTime": raw.get("processingTime", "N/A"),
    }


def analyze_text(text: str) -> tuple[bool, Dict[str, Any] | None, str | None]:
    """
    Analyze text for sarcasm using both models.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Tuple of (success, results_dict, error_message)
    """
    if not text.strip():
        return False, None, "Please enter some text to analyze"

    error = validate_input(text)
    if error:
        return False, None, error

    baseline_raw = backend_app.predict_baseline(text)
    proposed_raw = backend_app.predict_proposed(text)

    baseline_error = baseline_raw.get("error") if isinstance(baseline_raw, dict) else None
    if baseline_error:
        return False, None, f"Baseline model error: {baseline_error}"

    results = {
        "baseline": format_model_result(baseline_raw, "baseline"),
        "proposed": format_model_result(proposed_raw, "proposed"),
    }

    proposed_error = proposed_raw.get("error") if isinstance(proposed_raw, dict) else None
    if proposed_error:
        return True, results, f"Proposed model error: {proposed_error}"

    return True, results, None
