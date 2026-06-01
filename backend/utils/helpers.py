"""Helper utility functions."""

import gc
import re
from typing import Any, Dict, List

import app as backend_app
import streamlit as st

from config import BATCH_PROCESSING_SIZE, MAX_DATASET_SAMPLES


def init_session_state() -> None:
    """Initialize Streamlit session state with default values."""
    from config import SESSION_STATE_DEFAULTS
    
    for key, value in SESSION_STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_word_count(text: str) -> int:
    """
    Count words in text.
    
    Args:
        text: Text to count words in
        
    Returns:
        Word count
    """
    return len([w for w in re.split(r"\s+", text.strip()) if w]) if text.strip() else 0


def process_dataset(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process dataset through both models.
    
    Args:
        dataset: List of dataset items with 'id', 'text', 'label'
        
    Returns:
        List of prediction results
    """
    if not dataset:
        return []

    # Limit dataset size to prevent memory exhaustion
    max_samples = MAX_DATASET_SAMPLES
    if len(dataset) > max_samples:
        st.warning(f"Dataset limited to {max_samples} samples (original: {len(dataset)}) to prevent resource exhaustion.")
        dataset = dataset[:max_samples]

    results: List[Dict[str, Any]] = []
    progress = st.progress(0)
    status = st.empty()

    total = len(dataset)
    batch_size = BATCH_PROCESSING_SIZE

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = dataset[start:end]

        for item in batch:
            baseline_result = backend_app.predict_baseline(item["text"])
            proposed_result = backend_app.predict_proposed(item["text"])

            # Handle errors from models
            if "error" in baseline_result:
                baseline_result = {"isSarcastic": False, "confidence": 0.0}
            if "error" in proposed_result:
                proposed_result = {"isSarcastic": False, "confidence": 0.0}

            actual_label = item["label"] if item["label"] is not None else bool(baseline_result.get("isSarcastic", False))

            results.append(
                {
                    **item,
                    "label": actual_label,
                    "baseline": {
                        "predicted": bool(baseline_result.get("isSarcastic", False)),
                        "confidence": float(baseline_result.get("confidence", 0.0)),
                        "correct": bool(baseline_result.get("isSarcastic", False)) == actual_label,
                    },
                    "proposed": {
                        "predicted": bool(proposed_result.get("isSarcastic", False)),
                        "confidence": float(proposed_result.get("confidence", 0.0)),
                        "correct": bool(proposed_result.get("isSarcastic", False)) == actual_label,
                    },
                }
            )

        progress_value = min(end / total, 1.0)
        progress.progress(progress_value)
        status.info(f"Processing... ({end}/{total})")
        gc.collect()

    progress.progress(1.0)
    status.success("Dataset processing complete.")
    
    return results
