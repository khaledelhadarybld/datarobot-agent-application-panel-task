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
"""Order extraction tool — parses user input to extract menu items and quantities."""

import json
import re

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# The restaurant menu (shared across tools)
# ---------------------------------------------------------------------------
MENU = {"pizza": 10, "burger": 8, "coke": 3}


@tool
def extract_order_items(user_input: str) -> str:
    """Extract food items and quantities from a user's order message.

    Scans the user's text for known menu items (pizza, burger, coke) and
    tries to find an associated quantity. If no quantity is mentioned for
    an item, it defaults to 1.

    Args:
        user_input: The raw text from the user describing their order.

    Returns:
        A JSON string with a list of {"item": ..., "quantity": ...} dicts,
        or an error message if nothing could be extracted.
    """
    # --- LOG INPUT ---
    print(f"[extract_order_items] INPUT: {user_input}")

    user_lower = user_input.lower()
    extracted_items: list[dict[str, object]] = []

    # For each menu item, look for patterns like "2 pizzas", "3 burgers", etc.
    for item_name in MENU:
        # Match patterns: "2 pizza", "two pizzas", or just "pizza"
        # We look for a number before the item name
        pattern = rf"(\d+)\s+{item_name}s?"
        match = re.search(pattern, user_lower)
        if match:
            quantity = int(match.group(1))
            extracted_items.append({"item": item_name, "quantity": quantity})
        elif item_name in user_lower:
            # Item mentioned without a number → default to 1
            extracted_items.append({"item": item_name, "quantity": 1})

    # Build the result
    if extracted_items:
        result = json.dumps({"items": extracted_items, "raw_input": user_input})
    else:
        result = json.dumps(
            {
                "items": [],
                "raw_input": user_input,
                "error": "No menu items found in the order.",
            }
        )

    # --- LOG OUTPUT ---
    print(f"[extract_order_items] OUTPUT: {result}")
    return result
