import json
from json_repair import repair_json
import logging

logger = logging.getLogger(__name__)


def parse_openai_arguments(raw_argument_str: str) -> tuple[str, dict]:
    """
    Parse and clean the `function.arguments` string from OpenAI function call response.

    Args:
        raw_argument_str (str): Raw string from function.arguments field.

    Returns:
        tuple[str, dict]: (cleaned string, parsed dict object or {})
    """
    original = raw_argument_str.strip()

    # Remove Markdown fencing like ```json or ```
    if original.startswith("```json"):
        original = original[len("```json"):].strip()
    elif original.startswith("```"):
        original = original[len("```"):].strip()
    if original.endswith("```"):
        original = original[:-3].strip()

    # Replace some common LLM-generated issues
    cleaned = (
        original
        .replace("\\n", " ")
        .replace("\\", "")
        .replace("{{", "{")
        .replace("}}", "}")
        .replace("'", '"')  # optional: only if model returns single quotes
    ).strip()

    # Try normal JSON parsing
    try:
        result = json.loads(cleaned)
        return cleaned, result
    except json.JSONDecodeError:
        logger.warning("Standard JSON parsing failed, trying json_repair...")

    # Try repairing broken JSON
    try:
        repaired = repair_json(cleaned)
        result = json.loads(repaired)
        return repaired, result
    except Exception as e:
        logger.error(f"Failed to repair and parse JSON: {e}")
        return cleaned, {}
