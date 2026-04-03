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
"""Order pricing tool — calculates total price based on the menu."""

import json

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# The restaurant menu (shared across tools)
# ---------------------------------------------------------------------------
MENU = {"pizza": 10, "burger": 8, "coke": 3}


@tool
def calculate_order_price(validated_items_json: str) -> str:
    """Calculate the total price for validated order items.

    Looks up each item's unit price from the menu, multiplies by quantity,
    and computes the grand total.

    Args:
        validated_items_json: A JSON string containing an "items" list,
            where each entry has "item" (str) and "quantity" (int).

    Returns:
        A JSON string with line-item prices and the grand total.
    """
    # --- LOG INPUT ---
    print(f"[calculate_order_price] INPUT: {validated_items_json}")

    try:
        data = json.loads(validated_items_json)
    except json.JSONDecodeError:
        result = json.dumps({"error": "Invalid JSON input."})
        print(f"[calculate_order_price] OUTPUT: {result}")
        return result

    items = data.get("items", [])
    line_items: list[dict[str, object]] = []
    grand_total = 0

    for entry in items:
        item_name = entry.get("item", "").lower()
        quantity = entry.get("quantity", 0)
        unit_price = MENU.get(item_name, 0)
        line_total = unit_price * quantity

        line_items.append(
            {
                "item": item_name,
                "quantity": quantity,
                "unit_price": unit_price,
                "line_total": line_total,
            }
        )
        grand_total += line_total

    result = json.dumps(
        {
            "line_items": line_items,
            "grand_total": grand_total,
        }
    )

    # --- LOG OUTPUT ---
    print(f"[calculate_order_price] OUTPUT: {result}")
    return result
