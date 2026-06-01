"""Input validation functions."""

import re
from typing import Optional


def validate_input(input_text: str) -> Optional[str]:
    """
    Validate user input text for sarcasm detection.
    
    Args:
        input_text: Text to validate
        
    Returns:
        Error message if validation fails, None if valid
    """
    from config import MAX_WORDS
    
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
    if word_count > MAX_WORDS:
        return f"Error: Input exceeds maximum length. You entered {word_count} words, but the limit is {MAX_WORDS} words."

    return None
