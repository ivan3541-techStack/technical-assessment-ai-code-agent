"""Utility functions for JSON extraction and parsing."""

import json
from typing import Optional, List, Dict, Any


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract first JSON object {...} from text and parse it.

    Args:
        text: Text potentially containing a JSON object

    Returns:
        Parsed dictionary or None if no valid JSON object found
    """
    start = text.find("{")
    if start == -1:
        return None

    brace_level = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            brace_level += 1
        elif ch == "}":
            brace_level -= 1
            if brace_level == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    return None
    return None


def extract_json_array(text: str) -> Optional[List[Any]]:
    """
    Extract JSON array from text, ignoring markdown code blocks.

    Args:
        text: Text potentially containing a JSON array

    Returns:
        Parsed list or None if no valid JSON array found
    """
    # Clean markdown code blocks
    text = text.strip()

    # Remove ```json or ```
    if text.startswith("```json"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    # Find first [ and last ]
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end <= start:
        return None

    snippet = text[start:end].strip()
    try:
        data = json.loads(snippet)
        if isinstance(data, list):
            return data
        return None
    except json.JSONDecodeError:
        return None


def sanitize_table_cell(text: str) -> str:
    """
    Sanitize text for use in markdown table cells.

    Args:
        text: Text to sanitize

    Returns:
        Sanitized text with pipe characters replaced
    """
    return text.replace("|", "/")

