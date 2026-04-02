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
"""Math calculator tool — safely evaluates mathematical expressions."""

import math
from typing import Any

import numpy as np
from langchain_core.tools import tool

_SAFE_MATH_GLOBALS: dict[str, Any] = {
    "__builtins__": {},
    # math functions
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "int": int,
    "float": float,
    # math module
    "pi": math.pi,
    "e": math.e,
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "degrees": math.degrees,
    "radians": math.radians,
    "exp": math.exp,
    "inf": math.inf,
    # numpy functions (prefixed with np_)
    "np_array": np.array,
    "np_mean": np.mean,
    "np_median": np.median,
    "np_std": np.std,
    "np_var": np.var,
    "np_sum": np.sum,
    "np_dot": np.dot,
    "np_cross": np.cross,
    "np_linalg_norm": np.linalg.norm,
    "np_linalg_det": np.linalg.det,
    "np_linalg_inv": np.linalg.inv,
    "np_linspace": np.linspace,
    "np_arange": np.arange,
}


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Uses Python's ``math`` library and ``numpy`` for calculations.
    Supports arithmetic, trigonometry, logarithms, statistics, and
    basic linear algebra.

    Numpy functions are available with the ``np_`` prefix, e.g.
    ``np_mean([1, 2, 3])`` or ``np_linalg_det(np_array([[1,2],[3,4]]))``.

    Args:
        expression: A mathematical expression to evaluate.

    Returns:
        The result of the expression as a string, or an error message.
    """
    try:
        result = eval(expression, _SAFE_MATH_GLOBALS, {})  # noqa: S307
        return f"Result: {result}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except (SyntaxError, NameError, TypeError, ValueError) as e:
        return f"Error evaluating expression: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"
