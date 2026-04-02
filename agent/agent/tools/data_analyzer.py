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
"""Data analysis tool — loads JSON data into a DataFrame and runs operations."""

import json
from typing import Optional

import pandas as pd
from langchain_core.tools import tool


@tool
def analyze_data(
    data: str, operation: str, column: Optional[str] = None, query: Optional[str] = None
) -> str:
    """Analyze JSON data using pandas DataFrame operations.

    Supported operations: describe, head, tail, info, shape, columns,
    dtypes, value_counts, correlation, sort, filter, groupby, mean,
    median, sum, min, max, nunique, sample.

    Args:
        data: JSON string representing the data (list of dicts or dict of lists).
        operation: The analysis operation to perform.
        column: Optional column name for column-specific operations.
        query: Optional query string for filter/groupby operations.

    Returns:
        String representation of the analysis result.
    """
    try:
        parsed = json.loads(data)
        df = pd.DataFrame(parsed)
    except (json.JSONDecodeError, ValueError) as e:
        return f"Error parsing data: {e}"

    try:
        op = operation.lower().strip()

        if op == "describe":
            return str(df.describe(include="all"))
        elif op == "head":
            return str(df.head(10))
        elif op == "tail":
            return str(df.tail(10))
        elif op == "info":
            return (
                f"Shape: {df.shape}\nColumns: {list(df.columns)}\nDtypes:\n{df.dtypes}"
            )
        elif op == "shape":
            return f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"
        elif op == "columns":
            return str(list(df.columns))
        elif op == "dtypes":
            return str(df.dtypes)
        elif op == "value_counts":
            if column and column in df.columns:
                return str(df[column].value_counts())
            return "Please specify a valid column for value_counts."
        elif op == "correlation":
            numeric_df = df.select_dtypes(include="number")
            if numeric_df.empty:
                return "No numeric columns found for correlation."
            return str(numeric_df.corr())
        elif op == "sort":
            if column and column in df.columns:
                return str(df.sort_values(by=column, ascending=False).head(20))
            return "Please specify a valid column for sorting."
        elif op == "filter":
            if query:
                return str(df.query(query))
            return "Please provide a query string for filtering."
        elif op == "groupby":
            if column and query:
                return str(df.groupby(column).agg(query))
            elif column:
                return str(df.groupby(column).size())
            return "Please specify a column for groupby."
        elif op in ("mean", "median", "sum", "min", "max"):
            numeric_df = df.select_dtypes(include="number")
            if numeric_df.empty:
                return "No numeric columns found."
            return str(getattr(numeric_df, op)())
        elif op == "nunique":
            return str(df.nunique())
        elif op == "sample":
            n = min(5, len(df))
            return str(df.sample(n))
        else:
            return f"Unknown operation: {operation}. Use describe, head, filter, groupby, etc."
    except Exception as e:
        return f"Error during analysis: {e}"
