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
"""Order validation tool — checks items exist in menu and enforces guardrails."""

import json

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# The restaurant menu (shared across tools)
# ---------------------------------------------------------------------------
MENU = {"pizza": 10, "burger": 8, "coke": 3}

# Maximum quantity allowed per item (guardrail)
MAX_QUANTITY_PER_ITEM = 10


@tool
def validate_order(extracted_items_json: str) -> str:
    """Validate extracted order items against the menu and quantity limits.

    Guardrails:
      - Each item must exist in the menu (pizza, burger, coke).
      - Each item's quantity must be between 1 and 10 (inclusive).

    Args:
        extracted_items_json: A JSON string containing an "items" list,
            where each entry has "item" (str) and "quantity" (int).

    Returns:
        A JSON string with validation results including an "is_valid"
        boolean and a list of any "errors" found.
    """
    # --- LOG INPUT ---
    print(f"[validate_order] INPUT: {extracted_items_json}")

    try:
        data = json.loads(extracted_items_json)
    except json.JSONDecodeError:
        result = json.dumps({"is_valid": False, "errors": ["Invalid JSON input."]})
        print(f"[validate_order] OUTPUT: {result}")
        return result

    items = data.get("items", [])
    errors: list[str] = []

    # If no items were extracted, the order is invalid
    if not items:
        error_msg = data.get("error", "No items in the order.")
        errors.append(error_msg)

    # Check each item against the menu and quantity limits
    for entry in items:
        item_name = entry.get("item", "").lower()
        quantity = entry.get("quantity", 0)

        # Check if item exists in the menu
        if item_name not in MENU:
            errors.append(
                f"'{item_name}' is not on the menu. "
                f"Available items: {', '.join(MENU.keys())}."
            )

        # Check quantity guardrail (must be 1–10)
        if quantity < 1:
            errors.append(
                f"Quantity for '{item_name}' must be at least 1 (got {quantity})."
            )
        elif quantity > MAX_QUANTITY_PER_ITEM:
            errors.append(
                f"Quantity for '{item_name}' exceeds maximum of "
                f"{MAX_QUANTITY_PER_ITEM} (got {quantity})."
            )

    is_valid = len(errors) == 0
    result = json.dumps(
        {
            "is_valid": is_valid,
            "items": items,
            "errors": errors,
        }
    )

    # --- LOG OUTPUT ---
    print(f"[validate_order] OUTPUT: {result}")
    return result
