"""JSON-from-LLM-response parsing for the Sonnet analyzer (S5.3 AC-3).

The LLM is instructed to emit a single JSON object with no surrounding
prose, but real-world models occasionally wrap the output in markdown code
fences (``` or ```json) or emit a leading/trailing whitespace narration.
This module is the single seam that strips fences and parses, so the
analyzer's main path stays focused on routing decisions.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Match a fenced block: optional language tag, content, closing fence.
# Greedy on the content so embedded triple-backticks (rare) don't truncate.
_FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$",
    flags=re.DOTALL,
)


class ResponseParseError(Exception):
    """Raised when the LLM response cannot be parsed into a JSON object."""


def extract_json_object(text: str) -> dict[str, Any]:
    """Return the JSON object encoded in `text`, stripping ```json fences.

    Raises `ResponseParseError` if the text isn't a valid single object.
    Lists / scalars / multiple objects are rejected — the schema contract
    is "exactly one JSON object" (architecture §3.2).
    """
    cleaned = text.strip()
    fenced = _FENCE_RE.match(cleaned)
    if fenced is not None:
        cleaned = fenced.group(1).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ResponseParseError(f"not valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise ResponseParseError(
            f"expected JSON object, got {type(parsed).__name__}"
        )
    return parsed
