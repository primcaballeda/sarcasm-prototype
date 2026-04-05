import csv
import io
import json
import os
import re
from typing import Any, Dict, List, Optional

import streamlit as st

import app as backend_app


CONFUSION_MATRIX_BASELINE = {
    "truePositive": 651,
    "falsePositive": 280,
    "falseNegative": 277,
    "trueNegative": 670,
}

CONFUSION_MATRIX_PROPOSED = {
    "truePositive": 681,
    "falsePositive": 196,
    "falseNegative": 258,
    "trueNegative": 743,
}

EXAMPLES = [
    "Oh great, another Monday morning meeting!",
    "Yeah right, like that's ever going to happen...",
    "I love working on weekends!",
    "Thank you for your help today.",
]


st.set_page_config(page_title="Sarcasm Detector", page_icon="SD", layout="wide")


def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
                max-width: 1200px;
            }
            .title-wrap {
                text-align: center;
                margin-bottom: 1rem;
            }
            .subtitle {
                color: #475569;
                text-align: center;
                margin-top: -8px;
                margin-bottom: 20px;
            }
            .card {
                border: 1px solid #dbe5f2;
                border-radius: 12px;
                padding: 18px;
                background: #ffffff;
                margin-bottom: 14px;
            }
            .status-ok {
                color: #166534;
                font-weight: 700;
            }
            .status-bad {
                color: #991b1b;
                font-weight: 700;
            }
            .small-muted {
                color: #64748b;
                font-size: 0.9rem;
            }
            .pill {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 6px;
                background: #f1f5f9;
                color: #0f172a;
                font-size: 0.8rem;
                margin-right: 6px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    defaults = {
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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def validate_input(input_text: str) -> Optional[str]:
    trimmed_text = input_text.strip()

    if re.fullmatch(r"\d+", trimmed_text):
        return "Error: Please enter meaningful text, not just numbers."

    if re.fullmatch(r"-\d+", trimmed_text):
        return "Error: Negative numbers are not valid input. Please enter actual text."

    letter_count = len(re.findall(r"[a-zA-Z]", trimmed_text))
    total_chars = len(trimmed_text)
    if total_chars > 0 and (letter_count / total_chars) < 0.3:
        return "Error: Input appears to be random characters or special symbols. Please enter meaningful text."

    word_count = len([word for word in re.split(r"\s+", trimmed_text) if word])
    if word_count > 200:
        return f"Error: Input exceeds maximum length. You entered {word_count} words, but the limit is 200 words."

    return None


def safe_percentage(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "0.00"


def format_model_result(raw: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    if model_name == "baseline":
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
            "model": "GloVe+CNN+BiLSTM+Attention",
            "processingTime": raw.get("processingTime", "N/A"),
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
        "model": "BERT+CNN+BiLSTM+MHA",
        "processingTime": raw.get("processingTime", "N/A"),
    }


def parse_label(label_value: Any) -> Optional[bool]:
    if label_value is None:
        return None

    text = str(label_value).strip().lower()
    mapping = {
        "sarc": True,
        "sarcastic": True,
        "1": True,
        "true": True,
        "notsarc": False,
        "not sarcastic": False,
        "0": False,
        "false": False,
    }
    return mapping.get(text)


def parse_json_dataset(content: str) -> List[Dict[str, Any]]:
    json_data = json.loads(content)
    json_array = json_data if isinstance(json_data, list) else [json_data]

    parsed_data = []
    for index, item in enumerate(json_array):
        candidate_text = None
        if isinstance(item, dict):
            candidate_text = (
                item.get("text")
                or item.get("comment")
                or item.get("sentence")
                or item.get("Response Text")
                or item.get("response")
                or item.get("content")
            )

        if not candidate_text or not str(candidate_text).strip():
            continue

        label = parse_label(item.get("label") if isinstance(item, dict) else None)
        if label is None and isinstance(item, dict):
            label = parse_label(item.get("sarcastic"))
        if label is None and isinstance(item, dict):
            label = parse_label(item.get("is_sarcastic"))
        if label is None and isinstance(item, dict):
            label = parse_label(item.get("Label"))

        parsed_data.append(
            {
                "id": index + 1,
                "text": str(candidate_text).strip(),
                "label": label,
            }
        )

    if not parsed_data:
        raise ValueError(
            "JSON format is not aligned. Add a text field named text, comment, sentence, response, content, or Response Text."
        )

    return parsed_data


def parse_csv_dataset(content: str) -> List[Dict[str, Any]]:
    expected_headers = ["corpus", "label", "id", "response text"]
    normalized_content = content.replace("\r\n", "\n").replace("\r", "\n")

    reader = csv.reader(io.StringIO(normalized_content))
    rows = [row for row in reader]

    if len(rows) < 2:
        raise ValueError("CSV file must have at least a header row and one data row")

    headers = [h.strip().lower().strip('"') for h in rows[0]]
    if headers != expected_headers:
        raise ValueError("CSV format is not aligned. Expected exact header order: Corpus,Label,ID,Response Text")

    parsed_data = []
    for idx, row in enumerate(rows[1:], start=2):
        if len(row) != 4:
            raise ValueError(f"Row {idx} has {len(row)} columns. Expected exactly 4 columns.")

        normalized_values = [value.strip() for value in row]
        normalized_lower_values = [value.lower() for value in normalized_values]
        if normalized_lower_values == expected_headers:
            raise ValueError(f"Row {idx} appears to be a duplicate header row. Remove extra headers from the data section.")

        corpus, label_value, original_id, text_value = normalized_values
        if not corpus or not label_value or not original_id or not text_value:
            raise ValueError(
                f"Row {idx} is incomplete. All columns (Corpus, Label, ID, Response Text) are required."
            )

        if label_value.lower() not in {"sarc", "notsarc"}:
            raise ValueError(f"Row {idx} has invalid Label '{label_value}'. Use only sarc or notsarc.")

        if not re.fullmatch(r"\d+", original_id):
            raise ValueError(f"Row {idx} has invalid ID '{original_id}'. ID must be a number.")

        parsed_data.append(
            {
                "id": original_id,
                "text": text_value,
                "label": True if label_value.lower() == "sarc" else False,
            }
        )

    if not parsed_data:
        raise ValueError("No valid data rows found in CSV. Please check the file format.")

    return parsed_data


def parse_uploaded_file(uploaded_file: Any) -> List[Dict[str, Any]]:
    name = uploaded_file.name.lower()
    content = uploaded_file.getvalue().decode("utf-8")

    if name.endswith(".json"):
        return parse_json_dataset(content)
    if name.endswith(".csv"):
        return parse_csv_dataset(content)

    raise ValueError("Unsupported file type. Please upload a CSV or JSON file.")


def calculate_dataset_stats(dataset_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
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
    baseline_metrics = None
    proposed_metrics = None

    baseline_path = os.path.join("model", "model_metrics.json")
    proposed_path = os.path.join("model", "proposed_model_metrics.json")

    if os.path.exists(baseline_path):
        with open(baseline_path, "r", encoding="utf-8") as file:
            baseline_metrics = json.load(file)

    if os.path.exists(proposed_path):
        with open(proposed_path, "r", encoding="utf-8") as file:
            proposed_metrics = json.load(file)

    return {"baseline": baseline_metrics, "proposed": proposed_metrics}


def build_performance_rows(model_metrics: Dict[str, Any]) -> List[Dict[str, str]]:
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
                "Baseline (%)": safe_percentage((baseline_value or 0) * 100),
                "Proposed (%)": safe_percentage((proposed_value or 0) * 100),
            }
        )

    return rows


def analyze_text(text: str) -> Optional[str]:
    if not text.strip():
        return None

    error = validate_input(text)
    if error:
        return error

    baseline_raw = backend_app.predict_baseline(text)
    proposed_raw = backend_app.predict_proposed(text)

    st.session_state["results"] = {
        "baseline": format_model_result(baseline_raw, "baseline"),
        "proposed": format_model_result(proposed_raw, "proposed"),
    }
    return None


def process_dataset() -> None:
    dataset = st.session_state["dataset"]
    if not dataset:
        return

    st.session_state["dataset_results"] = []
    st.session_state["show_all_results"] = False

    progress = st.progress(0)
    status = st.empty()

    results: List[Dict[str, Any]] = []
    batch_size = 10
    total = len(dataset)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = dataset[start:end]

        for item in batch:
            baseline_result = backend_app.predict_baseline(item["text"])
            proposed_result = backend_app.predict_proposed(item["text"])

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

        st.session_state["dataset_results"] = results
        progress_value = min(end / total, 1.0)
        progress.progress(progress_value)
        status.info(f"Processing... ({end}/{total})")

    progress.progress(1.0)
    status.success("Dataset processing complete.")


def render_single_result(title: str, model_type: str, result: Dict[str, Any]) -> None:
    with st.container(border=True):
        status_text = "SARCASM DETECTED!" if result["isSarcastic"] else "Not Sarcastic"
        status_class = "status-bad" if result["isSarcastic"] else "status-ok"

        st.markdown(f"### {title}")
        st.markdown(f"<div class='{status_class}'>{status_text}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>{model_type}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>Processing time: {result['processingTime']}</div>", unsafe_allow_html=True)

        st.progress(min(max(result["confidence"] / 100.0, 0.0), 1.0), text=f"Confidence: {result['confidence']:.2f}%")

        st.markdown("Analysis Details")
        for indicator in result["indicators"]:
            st.write(f"- {indicator}")


def render_metrics_table(detailed_metrics: Dict[str, Any]) -> None:
    rows = []
    for metric_name, key in [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Sensitivity", "recall"),
        ("F1-Score", "f1Score"),
        ("Specificity", "specificity"),
    ]:
        baseline_value = float(detailed_metrics["baseline"][key])
        proposed_value = float(detailed_metrics["proposed"][key])
        improvement = proposed_value - baseline_value
        sign = "+" if improvement > 0 else ""

        rows.append(
            {
                "Metric": metric_name,
                "Baseline Model": f"{baseline_value:.2f}%",
                "Proposed Model": f"{proposed_value:.2f}%",
                "Improvement": f"{sign}{improvement:.2f}%",
            }
        )

    st.table(rows)


def render_confusion(conf: Dict[str, int], title: str) -> None:
    st.markdown(f"##### {title}")
    st.markdown(
        f"""
|  | Pred: Sarc | Pred: Not Sarc |
|---|---:|---:|
| Actual: Sarc | TP: {conf['tp']} | FN: {conf['fn']} |
| Actual: Not Sarc | FP: {conf['fp']} | TN: {conf['tn']} |
        """
    )


def main() -> None:
    apply_custom_style()
    init_state()

    st.markdown("<div class='title-wrap'><h1>Sarcasm Detector</h1></div>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Baseline vs. Proposed Model (Python + Streamlit)</p>", unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("Text Analysis")

        text_value = st.text_area(
            "Enter text",
            value=st.session_state["text"],
            placeholder="Type or paste text to analyze... (e.g., 'Oh great, another meeting!') [Max 200 words]",
            height=150,
        )
        st.session_state["text"] = text_value

        word_count = len([w for w in re.split(r"\s+", text_value.strip()) if w]) if text_value.strip() else 0
        if word_count > 200:
            st.error(f"{word_count} / 200 words")
        else:
            st.caption(f"{word_count} / 200 words")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            analyze_clicked = st.button("Detect Sarcasm", use_container_width=True)
        with col_b:
            clear_clicked = st.button("Clear", use_container_width=True)

        if analyze_clicked:
            message = analyze_text(st.session_state["text"])
            if message:
                st.error(message)

        if clear_clicked:
            st.session_state["text"] = ""
            st.session_state["results"] = None
            st.rerun()

    with st.container(border=True):
        st.subheader("Try These Examples")
        cols = st.columns(4)
        for index, sample in enumerate(EXAMPLES):
            with cols[index]:
                if st.button(f"Example {index + 1}", key=f"example_{index}", use_container_width=True):
                    st.session_state["text"] = sample
                    st.rerun()

    if st.session_state["results"]:
        st.subheader("Model Comparison Results")
        left, right = st.columns(2)
        with left:
            render_single_result(
                "Baseline Model",
                "GloVe + CNN + BiLSTM + Attention",
                st.session_state["results"]["baseline"],
            )
        with right:
            render_single_result(
                "Proposed Model",
                "BERT + CNN + BiLSTM + MHA",
                st.session_state["results"]["proposed"],
            )

    st.subheader("Upload Dataset for Batch Testing")
    st.caption("Use the guided steps below to compare both models on many text samples at once.")

    uploaded_file = st.file_uploader("Choose Dataset File", type=["csv", "json"])

    if uploaded_file is not None:
        signature = (uploaded_file.name, uploaded_file.size)
        if signature != st.session_state["uploaded_signature"]:
            st.session_state["uploaded_signature"] = signature
            try:
                parsed_data = parse_uploaded_file(uploaded_file)
                st.session_state["dataset"] = parsed_data
                st.session_state["dataset_results"] = []
                st.session_state["show_all_results"] = False
                st.session_state["upload_status"] = {
                    "type": "success",
                    "message": f"Loaded {len(parsed_data)} sample{'s' if len(parsed_data) != 1 else ''}. Click Process Dataset to run both models.",
                    "fileName": uploaded_file.name,
                }
            except Exception as exc:
                st.session_state["dataset"] = []
                st.session_state["dataset_results"] = []
                st.session_state["show_all_results"] = False
                st.session_state["upload_status"] = {
                    "type": "error",
                    "message": f"File format not aligned: {exc}",
                    "fileName": uploaded_file.name,
                }

    upload_status = st.session_state["upload_status"]
    if upload_status["type"] == "success":
        st.success(f"Status: {upload_status['message']} File: {upload_status['fileName']}")
    elif upload_status["type"] == "error":
        st.error(f"Status: {upload_status['message']} File: {upload_status['fileName']}")
    else:
        st.info(upload_status["message"])

    if st.session_state["dataset"]:
        st.write(f"{len(st.session_state['dataset'])} samples loaded")
        col_c, col_d = st.columns([1, 1])
        with col_c:
            if st.button("Process Dataset", use_container_width=True):
                process_dataset()
        with col_d:
            if st.button("Clear Dataset", use_container_width=True):
                st.session_state["dataset"] = []
                st.session_state["dataset_results"] = []
                st.session_state["show_all_results"] = False
                st.session_state["upload_status"] = {
                    "type": "neutral",
                    "message": "Dataset cleared. Upload a CSV or JSON file to start again.",
                    "fileName": "",
                }
                st.session_state["uploaded_signature"] = None
                st.rerun()

    dataset_results = st.session_state["dataset_results"]
    if dataset_results:
        st.subheader("Dataset Results")

        stats = calculate_dataset_stats(dataset_results)
        detailed_metrics = calculate_detailed_metrics(dataset_results)

        if detailed_metrics:
            st.markdown("#### Performance on Your Dataset")
            render_metrics_table(detailed_metrics)

            left_conf, right_conf = st.columns(2)
            with left_conf:
                render_confusion(detailed_metrics["baseline"]["confusion"], "Baseline Model - Confusion Matrix")
            with right_conf:
                render_confusion(detailed_metrics["proposed"]["confusion"], "Proposed Model - Confusion Matrix")

        if stats:
            st.markdown("#### Baseline Model Results")
            cols_stats_baseline = st.columns(5)
            cols_stats_baseline[0].metric("Total Samples", stats["total"])
            cols_stats_baseline[1].metric("Predicted Sarcastic", stats["baseline"]["predictedSarcastic"])
            cols_stats_baseline[2].metric("Predicted Not Sarcastic", stats["baseline"]["predictedNotSarcastic"])
            cols_stats_baseline[3].metric("Correct", f"{stats['baseline']['correct']}/{stats['withLabels']}")
            cols_stats_baseline[4].metric("Accuracy", f"{stats['baseline']['accuracy']}%")

            st.markdown("#### Proposed Model Results")
            cols_stats_proposed = st.columns(5)
            cols_stats_proposed[0].metric("Total Samples", stats["total"])
            cols_stats_proposed[1].metric("Predicted Sarcastic", stats["proposed"]["predictedSarcastic"])
            cols_stats_proposed[2].metric("Predicted Not Sarcastic", stats["proposed"]["predictedNotSarcastic"])
            cols_stats_proposed[3].metric("Correct", f"{stats['proposed']['correct']}/{stats['withLabels']}")
            cols_stats_proposed[4].metric("Accuracy", f"{stats['proposed']['accuracy']}%")

        show_all = st.checkbox(
            f"Show all results ({len(dataset_results)})",
            value=st.session_state["show_all_results"],
        )
        st.session_state["show_all_results"] = show_all
        display_rows = dataset_results if show_all else dataset_results[:15]

        has_labels = any(row.get("label") is not None for row in dataset_results)
        table_rows: List[Dict[str, Any]] = []
        for row in display_rows:
            output = {
                "ID": row["id"],
                "Text": row["text"],
                "Baseline Predicted": "Sarcastic" if row["baseline"]["predicted"] else "Not Sarcastic",
                "Baseline Confidence": f"{row['baseline']['confidence']:.2f}%",
                "Proposed Predicted": "Sarcastic" if row["proposed"]["predicted"] else "Not Sarcastic",
                "Proposed Confidence": f"{row['proposed']['confidence']:.2f}%",
            }
            if has_labels:
                output["Actual Label"] = "Sarcastic" if row.get("label") else "Not Sarcastic"
                output["Baseline Match"] = "PASS" if row["baseline"]["correct"] else "FAIL"
                output["Proposed Match"] = "PASS" if row["proposed"]["correct"] else "FAIL"
            table_rows.append(output)

        st.dataframe(table_rows, use_container_width=True)

    st.subheader("Model Performance Comparison")
    metrics = load_model_metrics()
    if not metrics.get("baseline") and not metrics.get("proposed"):
        st.warning(
            "Could not load metrics. Ensure model_metrics.json and proposed_model_metrics.json exist in backend/model/."
        )
    else:
        perf_rows = build_performance_rows(metrics)
        st.table(perf_rows)

        chart_source = [
            {
                "Metric": row["Metric"],
                "Baseline": float(row["Baseline (%)"]),
                "Proposed": float(row["Proposed (%)"]),
            }
            for row in perf_rows
        ]
        st.bar_chart(chart_source, x="Metric", y=["Baseline", "Proposed"])

    st.subheader("Confusion Matrix - Baseline Model")
    render_confusion(
        {
            "tp": CONFUSION_MATRIX_BASELINE["truePositive"],
            "tn": CONFUSION_MATRIX_BASELINE["trueNegative"],
            "fp": CONFUSION_MATRIX_BASELINE["falsePositive"],
            "fn": CONFUSION_MATRIX_BASELINE["falseNegative"],
        },
        "Reference Test Set",
    )

    st.subheader("Confusion Matrix - Proposed Model")
    render_confusion(
        {
            "tp": CONFUSION_MATRIX_PROPOSED["truePositive"],
            "tn": CONFUSION_MATRIX_PROPOSED["trueNegative"],
            "fp": CONFUSION_MATRIX_PROPOSED["falsePositive"],
            "fn": CONFUSION_MATRIX_PROPOSED["falseNegative"],
        },
        "Reference Test Set",
    )


if __name__ == "__main__":
    main()
