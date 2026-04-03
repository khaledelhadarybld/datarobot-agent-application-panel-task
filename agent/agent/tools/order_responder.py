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
"""Order response formatter tool — builds the final user-facing message."""

import json

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# The restaurant menu (for reference in error messages)
# ---------------------------------------------------------------------------
MENU = {"pizza": 10, "burger": 8, "coke": 3}


@tool
def format_order_response(order_data_json: str) -> str:
    """Format the final order response for the user.

    Produces a friendly confirmation message for valid orders, or a
    helpful error message for invalid ones.

    Args:
        order_data_json: A JSON string containing order processing results.
            For valid orders: includes "confirmed", "message", "grand_total".
            For invalid orders: includes "is_valid" (False) and "errors".

    Returns:
        A human-readable string with the order result.
    """
    # --- LOG INPUT ---
    print(f"[format_order_response] INPUT: {order_data_json}")

    try:
        data = json.loads(order_data_json)
    except json.JSONDecodeError:
        result = "Sorry, something went wrong processing your order. Please try again."
        print(f"[format_order_response] OUTPUT: {result}")
        return result

    # Check if this is a valid, confirmed order
    if data.get("confirmed", False):
        result = (
            f"🎉 {data.get('message', 'Order confirmed!')}\n\n"
            f"Thank you for your order! Your food is being prepared."
        )
    else:
        # Invalid order — show errors and the menu
        errors = data.get("errors", ["Unknown error."])
        error_list = "\n".join(f"  ❌ {e}" for e in errors)
        menu_display = "\n".join(
            f"  • {name}: ${price}" for name, price in MENU.items()
        )
        result = (
            f"⚠️ Sorry, there were issues with your order:\n"
            f"{error_list}\n\n"
            f"📋 Our menu:\n{menu_display}\n\n"
            f"Please try again with valid items and quantities (max 10 per item)."
        )

    # --- LOG OUTPUT ---
    print(f"[format_order_response] OUTPUT: {result}")
    return result
