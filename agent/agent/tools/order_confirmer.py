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
"""Order confirmation tool — simulates human-in-the-loop confirmation."""

import json

from langchain_core.tools import tool


@tool
def confirm_order(pricing_json: str) -> str:
    """Simulate a human-in-the-loop confirmation step for the order.

    In a real system this would pause and wait for a human to approve.
    Here we auto-approve valid orders to demonstrate the pattern.

    Args:
        pricing_json: A JSON string with "line_items" and "grand_total"
            from the pricing step.

    Returns:
        A JSON string with "confirmed" (bool) and a summary message.
    """
    # --- LOG INPUT ---
    print(f"[confirm_order] INPUT: {pricing_json}")

    try:
        data = json.loads(pricing_json)
    except json.JSONDecodeError:
        result = json.dumps(
            {
                "confirmed": False,
                "message": "Could not parse pricing data.",
            }
        )
        print(f"[confirm_order] OUTPUT: {result}")
        return result

    grand_total = data.get("grand_total", 0)
    line_items = data.get("line_items", [])

    # Build a human-readable summary of the order
    summary_lines: list[str] = []
    for li in line_items:
        summary_lines.append(
            f"  {li['quantity']}x {li['item']} "
            f"@ ${li['unit_price']} each = ${li['line_total']}"
        )
    summary = "\n".join(summary_lines)

    # --- Simulated human-in-the-loop: auto-approve ---
    # In production, you could integrate a Slack message, email, or UI prompt
    # that waits for a human to click "Approve" or "Reject".
    confirmed = True
    message = f"Order confirmed! ✅\nItems:\n{summary}\nGrand Total: ${grand_total}"

    result = json.dumps(
        {
            "confirmed": confirmed,
            "message": message,
            "grand_total": grand_total,
        }
    )

    # --- LOG OUTPUT ---
    print(f"[confirm_order] OUTPUT: {result}")
    return result
