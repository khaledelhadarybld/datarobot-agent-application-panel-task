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
"""Chart generation tool — creates charts from JSON data."""

import base64
import io
import json
from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt
from langchain_core.tools import tool

matplotlib.use("Agg")


def _parse_data(data_str: str) -> dict[str, Any]:
    """Parse a JSON string into a dictionary."""
    result: dict[str, Any] = json.loads(data_str)
    return result


@tool
def generate_chart(
    data: str,
    chart_type: str = "bar",
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
) -> str:
    """Generate a chart from JSON data and return it as a base64 PNG image.

    Supported chart types: bar, line, pie, scatter, histogram.

    Args:
        data: JSON string with the data. For bar/line/scatter provide
            ``{"x": [...], "y": [...]}``.  For pie provide
            ``{"labels": [...], "values": [...]}``.  For histogram
            provide ``{"values": [...]}``.
        chart_type: One of bar, line, pie, scatter, histogram.
        title: Optional chart title.
        x_label: Optional x-axis label.
        y_label: Optional y-axis label.

    Returns:
        A base64-encoded PNG image string, or an error message.
    """
    try:
        parsed = _parse_data(data)
    except json.JSONDecodeError as e:
        return f"Error parsing data: {e}"

    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ct = chart_type.lower().strip()

        if ct == "bar":
            ax.bar(parsed["x"], parsed["y"])
        elif ct == "line":
            ax.plot(parsed["x"], parsed["y"], marker="o")
        elif ct == "scatter":
            ax.scatter(parsed["x"], parsed["y"])
        elif ct == "pie":
            ax.pie(parsed["values"], labels=parsed.get("labels"), autopct="%1.1f%%")
        elif ct == "histogram":
            ax.hist(parsed["values"], bins="auto", edgecolor="black")
        else:
            plt.close(fig)
            return f"Unsupported chart type: {chart_type}. Use bar, line, pie, scatter, or histogram."

        if title:
            ax.set_title(title)
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"![chart](data:image/png;base64,{b64})"
    except KeyError as e:
        plt.close(fig)
        return f"Missing required data key: {e}"
    except Exception as e:
        plt.close(fig)
        return f"Error generating chart: {e}"
