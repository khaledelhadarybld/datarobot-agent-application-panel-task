# Copyright 2025 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PII removal tool — detects and redacts personally identifiable information."""

import re

from langchain_core.tools import tool

_PII_PATTERNS: dict[str, str] = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b",
    "ssn": r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "date_of_birth": r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
}


@tool
def remove_pii(text: str) -> str:
    """Detect and redact personally identifiable information (PII) from text.

    Scans the input for emails, phone numbers, SSNs, credit card numbers,
    IP addresses, and dates of birth, replacing each match with a
    ``[REDACTED_<TYPE>]`` placeholder.

    Args:
        text: The text to scan and redact.

    Returns:
        The redacted text with a summary of what was found.
    """
    results: dict[str, int] = {}
    redacted = text

    for pii_type, pattern in _PII_PATTERNS.items():
        matches = re.findall(pattern, redacted)
        if matches:
            results[pii_type] = len(matches)
            redacted = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", redacted)

    if not results:
        return f"No PII detected in the provided text.\n\nOriginal text:\n{text}"

    summary_lines = [
        f"- {pii_type}: {count} occurrence(s)" for pii_type, count in results.items()
    ]
    summary = "\n".join(summary_lines)
    return f"PII detected and redacted:\n{summary}\n\nRedacted text:\n{redacted}"
