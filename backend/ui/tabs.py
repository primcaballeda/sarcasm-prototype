"""Tab rendering functions for the Streamlit app."""

from typing import Any, Dict, List

import streamlit as st

from config import EXAMPLES, BASELINE_MODEL_NAME, PROPOSED_MODEL_NAME
from core.prediction import analyze_text
from core.metrics import (
    calculate_dataset_stats,
    calculate_detailed_metrics,
    load_model_metrics,
    build_performance_rows,
)
from core.parsers import parse_uploaded_file
from ui.charts import (
    render_performance_comparison_chart,
    render_confusion_matrix,
    render_batch_results_gauges,
)
from utils.helpers import get_word_count, process_dataset
import app as backend_app


def render_predict_tab() -> None:
    """Render the 'Predict' tab for single text analysis."""
    st.subheader("Real-Time Sarcasm Detection")
    st.caption("Enter text to detect sarcasm using both model architectures")

    with st.container(border=True):
        text_value = st.text_area(
            "Enter text",
            value=st.session_state["text"],
            placeholder="Type or paste text to analyze... (e.g., 'Oh great, another meeting!') [Max 200 words]",
            height=150,
        )
        st.session_state["text"] = text_value

        if st.session_state["current_example"] is not None:
            st.info(f"Example {st.session_state['current_example']} loaded")

        word_count = get_word_count(text_value)
        if word_count > 200:
            st.error(f"[LIMIT EXCEEDED] {word_count} / 200 words - Text too long")
        else:
            st.caption(f"{word_count} / 200 words")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            analyze_clicked = st.button("Detect Sarcasm", width="stretch", key="predict_detect")
        with col_b:
            clear_clicked = st.button("Clear", width="stretch", key="predict_clear")

        if analyze_clicked:
            if not text_value.strip():
                st.warning("Please enter some text to analyze")
            else:
                status_widget = getattr(st, "status", None)
                if callable(status_widget):
                    with st.status("Detecting sarcasm...", expanded=False):
                        success, results, error_msg = analyze_text(text_value)
                else:
                    with st.spinner("Detecting sarcasm..."):
                        success, results, error_msg = analyze_text(text_value)
                
                if error_msg:
                    if error_msg.lower().startswith("proposed model error"):
                        st.warning(error_msg)
                    else:
                        st.error(error_msg)
                else:
                    st.session_state["results"] = results

        if clear_clicked:
            st.session_state["text"] = ""
            st.session_state["results"] = None
            st.session_state["current_example"] = None
            st.rerun()

    # Example buttons
    st.markdown("### Try These Examples")
    cols = st.columns(4)
    for index, sample in enumerate(EXAMPLES):
        with cols[index]:
            if st.button(f"Example {index + 1}", key=f"example_{index}", width="stretch", help=sample):
                st.session_state["text"] = sample
                st.session_state["current_example"] = index + 1
                st.rerun()

    # Results display
    if st.session_state["results"]:
        st.markdown("---")
        _render_prediction_results()


def _render_prediction_results() -> None:
    """Render prominent prediction results."""
    baseline_result = st.session_state["results"]["baseline"]
    proposed_result = st.session_state["results"]["proposed"]

    both_sarcastic = baseline_result["isSarcastic"] and proposed_result["isSarcastic"]
    both_not = not baseline_result["isSarcastic"] and not proposed_result["isSarcastic"]

    st.markdown("## PREDICTION RESULT")

    if both_sarcastic:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(244, 63, 94, 0.1) 0%, rgba(244, 114, 182, 0.08) 100%); border: 1px solid rgba(244, 63, 94, 0.3); border-left: 4px solid #f43f5e; padding: 32px; border-radius: 12px;">
            <h2 style="color: #f43f5e; margin: 0; font-size: 2.2rem; font-weight: 700;">SARCASM DETECTED</h2>
            <p style="color: #2d2d3d; margin: 12px 0 0 0; font-size: 14px; font-weight: 400;">Both models agree</p>
        </div>
        """, unsafe_allow_html=True)
    elif both_not:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(74, 222, 128, 0.08) 100%); border: 1px solid rgba(34, 197, 94, 0.3); border-left: 4px solid #22c55e; padding: 32px; border-radius: 12px;">
            <h2 style="color: #22c55e; margin: 0; font-size: 2.2rem; font-weight: 700;">NOT SARCASTIC</h2>
            <p style="color: #2d2d3d; margin: 12px 0 0 0; font-size: 14px; font-weight: 400;">Both models agree</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(79, 70, 229, 0.1) 0%, rgba(124, 58, 237, 0.08) 100%); border: 1px solid rgba(124, 58, 237, 0.3); border-left: 4px solid #7c3aed; padding: 32px; border-radius: 12px;">
            <h2 style="color: #7c3aed; margin: 0; font-size: 2.2rem; font-weight: 700;">DISAGREEMENT</h2>
            <p style="color: #2d2d3d; margin: 12px 0 0 0; font-size: 14px; font-weight: 400;">Models predict differently</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Individual Predictions")
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown(f"#### {BASELINE_MODEL_NAME}")
            result_text = "SARCASTIC" if baseline_result["isSarcastic"] else "NOT SARCASTIC"
            st.markdown(f"**{result_text}**")
            st.markdown(f"Confidence: {baseline_result['confidence']:.1f}%")
            st.progress(baseline_result["confidence"] / 100)
            st.caption(f"Time: {baseline_result.get('processingTime', 'N/A')} ms")

    with col2:
        with st.container(border=True):
            st.markdown(f"#### {PROPOSED_MODEL_NAME}")
            result_text = "SARCASTIC" if proposed_result["isSarcastic"] else "NOT SARCASTIC"
            st.markdown(f"**{result_text}**")
            st.markdown(f"Confidence: {proposed_result['confidence']:.1f}%")
            st.progress(proposed_result["confidence"] / 100)
            st.caption(f"Time: {proposed_result.get('processingTime', 'N/A')} ms")


def render_batch_testing_tab() -> None:
    """Render the 'Batch Testing' tab for dataset evaluation."""
    st.subheader("Dataset Batch Evaluation")
    st.caption("Upload CSV or JSON files to evaluate both models on multiple samples")

    with st.container(border=True):
        st.markdown("### Upload Dataset")

        with st.expander("Accepted File Formats", expanded=False):
            st.markdown("""
**CSV Format:**
- Required columns: `Corpus`, `Label`, `ID`, `Response Text`
- Label: `sarc` (sarcastic) or `notsarc` (not sarcastic)
- Example:
  ```
  Corpus,Label,ID,Response Text
  ignored,sarc,1,Oh great another Monday
  ignored,notsarc,2,Thank you for your help
  ```

**JSON Format:**
- Text field: `text`, `comment`, `sentence`, `response`, or `content`
- Label field: `label` or `sarcastic`
- Supported label values: `sarc`/`sarcastic`/`1`/`true` or `notsarc`/`not sarcastic`/`0`/`false`
            """)

        uploaded_file = st.file_uploader("Choose Dataset File", type=["csv", "json"], key="batch_uploader")

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
                        "message": f"Loaded {len(parsed_data)} sample{'s' if len(parsed_data) != 1 else ''}",
                        "fileName": uploaded_file.name,
                    }
                except Exception as exc:
                    st.session_state["dataset"] = []
                    st.session_state["dataset_results"] = []
                    st.session_state["show_all_results"] = False
                    st.session_state["upload_status"] = {
                        "type": "error",
                        "message": f"File format error: {exc}",
                        "fileName": uploaded_file.name,
                    }

        upload_status = st.session_state["upload_status"]
        if upload_status["type"] == "success":
            st.success(f"[SUCCESS] {upload_status['message']} - File: {upload_status['fileName']}")
        elif upload_status["type"] == "error":
            st.error(f"[ERROR] {upload_status['message']} - File: {upload_status['fileName']}")
        else:
            st.info(upload_status["message"])

    if st.session_state["dataset"]:
        with st.container(border=True):
            st.markdown(f"### Processing: {len(st.session_state['dataset'])} Samples Loaded")
            col_c, col_d = st.columns([1, 1])
            with col_c:
                if st.button("Process Dataset", width="stretch", key="batch_process"):
                    results = process_dataset(st.session_state["dataset"])
                    st.session_state["dataset_results"] = results
            with col_d:
                if st.button("Clear Dataset", width="stretch", key="batch_clear"):
                    st.session_state["dataset"] = []
                    st.session_state["dataset_results"] = []
                    st.session_state["show_all_results"] = False
                    st.session_state["upload_status"] = {
                        "type": "neutral",
                        "message": "Dataset cleared. Upload a new file to continue.",
                        "fileName": "",
                    }
                    st.session_state["uploaded_signature"] = None
                    st.rerun()

    dataset_results = st.session_state["dataset_results"]
    if dataset_results:
        st.markdown("---")
        st.markdown("### Results")

        # Summary metrics
        stats = calculate_dataset_stats(dataset_results)
        if stats:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", stats["total"])
            with col2:
                baseline_acc = stats["baseline"]["accuracy"]
                st.metric("GloVe Accuracy", f"{baseline_acc}%")
            with col3:
                proposed_acc = stats["proposed"]["accuracy"]
                st.metric("BERT Accuracy", f"{proposed_acc}%")

        # Gauges
        st.markdown("### Performance Summary")
        render_batch_results_gauges(dataset_results)

        # Confusion matrices
        st.markdown("---")
        st.markdown("### Confusion Matrices")
        detailed_metrics = calculate_detailed_metrics(dataset_results)
        if detailed_metrics:
            left_conf, right_conf = st.columns(2)
            with left_conf:
                render_confusion_matrix(
                    detailed_metrics["baseline"]["confusion"],
                    f"{BASELINE_MODEL_NAME}"
                )
            with right_conf:
                render_confusion_matrix(
                    detailed_metrics["proposed"]["confusion"],
                    f"{PROPOSED_MODEL_NAME}"
                )

        # Per-sample table
        st.markdown("---")
        st.markdown("### Per-Sample Results")
        show_all = st.checkbox(
            f"Show all results ({len(dataset_results)})",
            value=st.session_state["show_all_results"],
            key="batch_show_all"
        )
        st.session_state["show_all_results"] = show_all
        display_rows = dataset_results if show_all else dataset_results[:20]

        has_labels = any(row.get("label") is not None for row in dataset_results)
        table_rows: List[Dict[str, Any]] = []
        for row in display_rows:
            output = {
                "ID": row["id"],
                "Text": row["text"][:50] + "..." if len(row["text"]) > 50 else row["text"],
                "GloVe": "Sarc" if row["baseline"]["predicted"] else "Not",
                "GloVe %": f"{row['baseline']['confidence']:.0f}%",
                "BERT": "Sarc" if row["proposed"]["predicted"] else "Not",
                "BERT %": f"{row['proposed']['confidence']:.0f}%",
            }
            if has_labels:
                output["Actual"] = "Sarc" if row.get("label") else "Not"
                output["Match"] = "Yes" if (row["baseline"]["correct"] and row["proposed"]["correct"]) else "No"
            table_rows.append(output)

        st.dataframe(table_rows, use_container_width=True)


def render_analytics_tab() -> None:
    """Render the 'Model Analytics' tab for pre-trained model performance."""
    st.subheader("Model Performance Analysis")

    metrics = load_model_metrics()
    if not metrics.get("baseline") and not metrics.get("proposed"):
        st.error("Could not load model metrics. Ensure JSON files exist in backend/model/")
        return

    # Model specifications
    st.markdown("### Model Specifications")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### {BASELINE_MODEL_NAME}")
        baseline_info = metrics.get("baseline", {}).get("model_info", {})
        st.write(f"""
- Embeddings: GloVe (300-dim)
- Architecture: CNN -> BiLSTM -> Attention
- Epochs: {baseline_info.get('epochs', 'N/A')}
- Batch Size: {baseline_info.get('batch_size', 'N/A')}
- Optimizer: Adam (lr={baseline_info.get('learning_rate', 'N/A')})
- Vocab Size: {metrics.get('baseline', {}).get('dataset_info', {}).get('vocab_size', 'N/A'):,}
        """)

    with col2:
        st.markdown(f"#### {PROPOSED_MODEL_NAME}")
        proposed_info = metrics.get("proposed", {}).get("model_info", {})
        st.write(f"""
- Embeddings: BERT (768-dim, contextual)
- Architecture: BERT -> CNN -> BiLSTM -> Multi-Head Attn
- Epochs: {proposed_info.get('epochs', 'N/A')}
- Batch Size: {proposed_info.get('batch_size', 'N/A')}
- Optimizer: Adam (lr={proposed_info.get('learning_rate', 'N/A')})
- Attention Heads: {proposed_info.get('num_heads', 'N/A')}
        """)

    # Performance metrics
    st.markdown("---")
    st.markdown("### Performance Metrics (Test Set)")

    baseline_perf = metrics.get("baseline", {}).get("performance_metrics", {})
    proposed_perf = metrics.get("proposed", {}).get("performance_metrics", {})

    metrics_list = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall/Sensitivity", "sensitivity_recall"),
        ("Specificity", "specificity"),
        ("F1-Score", "f1_score")
    ]

    table_data = []
    for display_name, key_name in metrics_list:
        baseline_val = baseline_perf.get(key_name, 0) * 100
        proposed_val = proposed_perf.get(key_name, 0) * 100
        difference = proposed_val - baseline_val

        diff_str = f"{difference:+.2f}%" if difference != 0 else "0.00%"

        table_data.append({
            "Metric": display_name,
            "GloVe": f"{baseline_val:.2f}%",
            "BERT": f"{proposed_val:.2f}%",
            "Improvement": diff_str
        })

    import pandas as pd
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Confusion matrices
    st.markdown("---")
    st.markdown("### Confusion Matrices")

    baseline_cm = metrics.get("baseline", {}).get("confusion_matrix", {})
    proposed_cm = metrics.get("proposed", {}).get("confusion_matrix", {})

    col1, col2 = st.columns(2)
    with col1:
        render_confusion_matrix(
            {
                "tp": baseline_cm.get("true_positives", 0),
                "tn": baseline_cm.get("true_negatives", 0),
                "fp": baseline_cm.get("false_positives", 0),
                "fn": baseline_cm.get("false_negatives", 0),
            },
            BASELINE_MODEL_NAME,
        )
    with col2:
        render_confusion_matrix(
            {
                "tp": proposed_cm.get("true_positives", 0),
                "tn": proposed_cm.get("true_negatives", 0),
                "fp": proposed_cm.get("false_positives", 0),
                "fn": proposed_cm.get("false_negatives", 0),
            },
            PROPOSED_MODEL_NAME,
        )

    # Performance chart
    st.markdown("---")
    st.markdown("### Performance Metrics Chart")
    perf_rows = build_performance_rows(metrics)
    render_performance_comparison_chart(perf_rows)
