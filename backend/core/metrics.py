"""Metrics calculation functions for batch testing."""

import json
import os
from typing import Any, Dict, List, Optional


def calculate_dataset_stats(dataset_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Calculate basic statistics from dataset evaluation results.
    
    Args:
        dataset_results: List of prediction results for dataset samples
        
    Returns:
        Dictionary with statistics or None if no results
    """
    if not dataset_results:
        return None

    total = len(dataset_results)
    with_labels = [row for row in dataset_results if row.get("label") is not None]

    baseline_correct = sum(1 for row in with_labels if row["baseline"]["correct"])
    proposed_correct = sum(1 for row in with_labels if row["proposed"]["correct"])

    baseline_sarcastic = sum(1 for row in dataset_results if row["baseline"]["predicted"])
    proposed_sarcastic = sum(1 for row in dataset_results if row["proposed"]["predicted"])

    baseline_accuracy = f"{(baseline_correct / len(with_labels) * 100):.2f}" if with_labels else "N/A"
    proposed_accuracy = f"{(proposed_correct / len(with_labels) * 100):.2f}" if with_labels else "N/A"

    return {
        "total": total,
        "withLabels": len(with_labels),
        "baseline": {
            "correct": baseline_correct,
            "accuracy": baseline_accuracy,
            "predictedSarcastic": baseline_sarcastic,
            "predictedNotSarcastic": total - baseline_sarcastic,
        },
        "proposed": {
            "correct": proposed_correct,
            "accuracy": proposed_accuracy,
            "predictedSarcastic": proposed_sarcastic,
            "predictedNotSarcastic": total - proposed_sarcastic,
        },
    }


def calculate_detailed_metrics(dataset_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Calculate detailed performance metrics (accuracy, precision, recall, F1, specificity).
    
    Args:
        dataset_results: List of prediction results for dataset samples
        
    Returns:
        Dictionary with detailed metrics per model or None if no labeled data
    """
    with_labels = [row for row in dataset_results if row.get("label") is not None]
    if not with_labels:
        return None

    def confusion(model_key: str) -> Dict[str, int]:
        tp = sum(1 for row in with_labels if row["label"] is True and row[model_key]["predicted"] is True)
        tn = sum(1 for row in with_labels if row["label"] is False and row[model_key]["predicted"] is False)
        fp = sum(1 for row in with_labels if row["label"] is False and row[model_key]["predicted"] is True)
        fn = sum(1 for row in with_labels if row["label"] is True and row[model_key]["predicted"] is False)
        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    def metric_block(conf: Dict[str, int]) -> Dict[str, Any]:
        total = len(with_labels)
        accuracy = ((conf["tp"] + conf["tn"]) / total) * 100 if total else 0.0
        precision = (conf["tp"] / (conf["tp"] + conf["fp"]) * 100) if (conf["tp"] + conf["fp"]) else 0.0
        recall = (conf["tp"] / (conf["tp"] + conf["fn"]) * 100) if (conf["tp"] + conf["fn"]) else 0.0
        specificity = (conf["tn"] / (conf["tn"] + conf["fp"]) * 100) if (conf["tn"] + conf["fp"]) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        return {
            "accuracy": f"{accuracy:.2f}",
            "precision": f"{precision:.2f}",
            "recall": f"{recall:.2f}",
            "f1Score": f"{f1:.2f}",
            "specificity": f"{specificity:.2f}",
            "confusion": conf,
        }

    baseline_conf = confusion("baseline")
    proposed_conf = confusion("proposed")

    return {
        "baseline": metric_block(baseline_conf),
        "proposed": metric_block(proposed_conf),
    }


def load_model_metrics() -> Dict[str, Any]:
    """
    Load pre-trained model metrics from JSON files.
    
    Returns:
        Dictionary with baseline and proposed model metrics
    """
    from config import BASELINE_METRICS_PATH, PROPOSED_METRICS_PATH
    
    baseline_metrics = None
    proposed_metrics = None

    if os.path.exists(BASELINE_METRICS_PATH):
        with open(BASELINE_METRICS_PATH, "r", encoding="utf-8") as file:
            baseline_metrics = json.load(file)

    if os.path.exists(PROPOSED_METRICS_PATH):
        with open(PROPOSED_METRICS_PATH, "r", encoding="utf-8") as file:
            proposed_metrics = json.load(file)

    return {"baseline": baseline_metrics, "proposed": proposed_metrics}


def build_performance_rows(model_metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Build formatted table rows for performance metrics display.
    
    Args:
        model_metrics: Model metrics dictionary
        
    Returns:
        List of row dictionaries for metrics table
    """
    baseline = (model_metrics.get("baseline") or {}).get("performance_metrics") or {}
    proposed = (model_metrics.get("proposed") or {}).get("performance_metrics") or {}

    metric_specs = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Sensitivity", "sensitivity_recall"),
        ("F1-Score", "f1_score"),
        ("Specificity", "specificity"),
    ]

    rows = []
    for title, key in metric_specs:
        baseline_value = baseline.get(key)
        proposed_value = proposed.get(key)
        rows.append(
            {
                "Metric": title,
                "GloVe+CNN+BiLSTM+Attn (%)": _safe_percentage((baseline_value or 0) * 100),
                "BERT+CNN+BiLSTM+MHA (%)": _safe_percentage((proposed_value or 0) * 100),
            }
        )

    return rows


def _safe_percentage(value: Any) -> str:
    """
    Safely convert a value to percentage string format.
    
    Args:
        value: Value to convert
        
    Returns:
        Formatted percentage string
    """
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "0.00"
