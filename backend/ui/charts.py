"""Chart and visualization functions."""

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

try:
    import altair as alt
except Exception:
    alt = None

from config import BASELINE_MODEL_NAME, PROPOSED_MODEL_NAME, COLOR_PURPLE, COLOR_CYAN


def render_performance_comparison_chart(perf_rows: List[Dict[str, str]]) -> None:
    """
    Render bar chart comparing performance metrics between models.
    
    Args:
        perf_rows: List of metric rows with 'Metric' and model score columns
    """
    chart_values: List[Dict[str, Any]] = []
    for row in perf_rows:
        metric = row.get("Metric")
        if not metric:
            continue

        try:
            baseline_value = float(row.get("GloVe+CNN+BiLSTM+Attn (%)", 0.0))
        except (TypeError, ValueError):
            baseline_value = 0.0

        try:
            proposed_value = float(row.get("BERT+CNN+BiLSTM+MHA (%)", 0.0))
        except (TypeError, ValueError):
            proposed_value = 0.0

        chart_values.append({"Metric": metric, "Model": BASELINE_MODEL_NAME, "Value": baseline_value})
        chart_values.append({"Metric": metric, "Model": PROPOSED_MODEL_NAME, "Value": proposed_value})

    if not chart_values:
        st.info("No chart data available.")
        return

    if alt is None:
        wide = []
        for row in perf_rows:
            metric = row.get("Metric")
            if not metric:
                continue
            wide.append(
                {
                    "Metric": metric,
                    "GloVe+CNN": float(row.get("GloVe+CNN+BiLSTM+Attn (%)") or 0.0),
                    "BERT+CNN": float(row.get("BERT+CNN+BiLSTM+MHA (%)") or 0.0),
                }
            )
        st.bar_chart(wide, x="Metric", y=["GloVe+CNN", "BERT+CNN"])
        return

    metric_order = [row["Metric"] for row in perf_rows if row.get("Metric")]

    base = (
        alt.Chart(alt.Data(values=chart_values))
        .encode(
            x=alt.X("Metric:N", sort=metric_order, axis=alt.Axis(labelAngle=0, title=None)),
            xOffset=alt.XOffset("Model:N"),
            y=alt.Y("Value:Q", title="Score (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "Model:N",
                legend=alt.Legend(orient="top"),
                scale=alt.Scale(
                    domain=[BASELINE_MODEL_NAME, PROPOSED_MODEL_NAME],
                    range=[COLOR_PURPLE, COLOR_CYAN],
                ),
            ),
            tooltip=["Metric:N", "Model:N", alt.Tooltip("Value:Q", format=".2f", title="Score (%)")],
        )
    )

    bars = base.mark_bar(size=32, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    labels = base.mark_text(dy=-8, fontWeight="bold").encode(text=alt.Text("Value:Q", format=".0f"))
    chart = (bars + labels).properties(height=320).configure_view(strokeWidth=0)

    st.altair_chart(chart, use_container_width=True)


def render_confusion_matrix(conf: Dict[str, int], title: str) -> None:
    """
    Render confusion matrix as markdown table.
    
    Args:
        conf: Confusion matrix with 'tp', 'tn', 'fp', 'fn' keys
        title: Title for the confusion matrix
    """
    st.markdown(f"##### {title}")
    st.markdown(
        f"""
|  | Pred: Sarc | Pred: Not Sarc |
|---|---:|---:|
| Actual: Sarc | TP: {conf['tp']} | FN: {conf['fn']} |
| Actual: Not Sarc | FP: {conf['fp']} | TN: {conf['tn']} |
        """
    )


def render_batch_results_gauges(dataset_results: List[Dict[str, Any]]) -> None:
    """
    Render performance gauges for batch results.
    
    Shows: GloVe Accuracy, BERT Accuracy, Agreement %, Improvement %
    
    Args:
        dataset_results: List of prediction results
    """
    from core.metrics import calculate_dataset_stats, calculate_detailed_metrics
    
    stats = calculate_dataset_stats(dataset_results)
    detailed = calculate_detailed_metrics(dataset_results)
    
    if not stats or not detailed:
        st.info("Not enough labeled data to calculate metrics.")
        return

    # Extract values
    baseline_acc = float(stats["baseline"]["accuracy"]) if stats["baseline"]["accuracy"] != "N/A" else 0
    proposed_acc = float(stats["proposed"]["accuracy"]) if stats["proposed"]["accuracy"] != "N/A" else 0
    
    # Calculate agreement %
    total = len(dataset_results)
    agreement_count = sum(
        1 for row in dataset_results 
        if row["baseline"]["predicted"] == row["proposed"]["predicted"]
    )
    agreement_pct = (agreement_count / total * 100) if total > 0 else 0
    
    # Calculate improvement
    improvement = proposed_acc - baseline_acc

    # Display gauges in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="GloVe Accuracy",
            value=f"{baseline_acc:.1f}%",
            delta=None,
        )

    with col2:
        st.metric(
            label="BERT Accuracy",
            value=f"{proposed_acc:.1f}%",
            delta=f"{improvement:+.1f}%" if improvement != 0 else "0%",
        )

    with col3:
        st.metric(
            label="Model Agreement",
            value=f"{agreement_pct:.1f}%",
            delta=f"{total - agreement_count} disagreements",
        )

    with col4:
        improvement_display = f"{improvement:+.1f}%" if improvement != 0 else "0%"
        st.metric(
            label="BERT Improvement",
            value=improvement_display,
            delta=None,
        )
