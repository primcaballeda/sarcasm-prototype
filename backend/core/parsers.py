"""CSV and JSON dataset parsing functions."""

import csv
import io
import json
import re
from typing import Any, Dict, List, Optional


def parse_label(label_value: Any) -> Optional[bool]:
    """
    Parse a label value to boolean (True=sarcastic, False=not sarcastic).
    
    Args:
        label_value: Raw label value
        
    Returns:
        True if sarcastic, False if not, None if unparseable
    """
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
    """
    Parse JSON dataset file.
    
    Args:
        content: JSON file content as string
        
    Returns:
        List of parsed data items with 'id', 'text', 'label'
        
    Raises:
        ValueError: If JSON format is invalid
    """
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
    """
    Parse CSV dataset file.
    
    Args:
        content: CSV file content as string
        
    Returns:
        List of parsed data items with 'id', 'text', 'label'
        
    Raises:
        ValueError: If CSV format is invalid
    """
    from config import EXPECTED_CSV_HEADERS, EXPECTED_LABEL_VALUES
    
    normalized_content = content.replace("\r\n", "\n").replace("\r", "\n")

    reader = csv.reader(io.StringIO(normalized_content))
    rows = [row for row in reader]

    if len(rows) < 2:
        raise ValueError("CSV file must have at least a header row and one data row")

    headers = [h.strip().lower().strip('"') for h in rows[0]]
    if headers != EXPECTED_CSV_HEADERS:
        raise ValueError("CSV format is not aligned. Expected exact header order: Corpus,Label,ID,Response Text")

    parsed_data = []
    for idx, row in enumerate(rows[1:], start=2):
        if len(row) != 4:
            raise ValueError(f"Row {idx} has {len(row)} columns. Expected exactly 4 columns.")

        normalized_values = [value.strip() for value in row]
        normalized_lower_values = [value.lower() for value in normalized_values]
        if normalized_lower_values == EXPECTED_CSV_HEADERS:
            raise ValueError(f"Row {idx} appears to be a duplicate header row. Remove extra headers from the data section.")

        corpus, label_value, original_id, text_value = normalized_values
        if not corpus or not label_value or not original_id or not text_value:
            raise ValueError(
                f"Row {idx} is incomplete. All columns (Corpus, Label, ID, Response Text) are required."
            )

        if label_value.lower() not in EXPECTED_LABEL_VALUES:
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
    """
    Parse an uploaded file (CSV or JSON).
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        List of parsed data items
        
    Raises:
        ValueError: If file format is unsupported
    """
    name = uploaded_file.name.lower()
    content = uploaded_file.getvalue().decode("utf-8")

    if name.endswith(".json"):
        return parse_json_dataset(content)
    if name.endswith(".csv"):
        return parse_csv_dataset(content)

    raise ValueError("Unsupported file type. Please upload a CSV or JSON file.")
