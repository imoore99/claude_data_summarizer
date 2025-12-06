import os
from typing import Dict, Any
import json

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

# Load .env so ANTHROPIC_API_KEY is available
load_dotenv()

# Create Anthropic client (reads ANTHROPIC_API_KEY from environment)
client = Anthropic()


def summarize_with_claude(df_summary: Dict[str, Any]) -> str:

    # Call Claude with the dataset summary and return a short analysis.
    # Include chart suggestions if available.
    resp = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=400,
        messages=[
            {
                "role": "user",
                "content": (
                    "You are a data analyst. Here is a JSON summary of a dataset:\n\n"
                    f"{df_summary}\n\n"
                    "Return ONLY valid JSON with this structure:\n"
                    "{\n"
                    '  \"text\": \"...1–2 short paragraphs plus 3–5 bullet insights. Do NOT label paragraphs as \'Paragraph 1\' or \'Paragraph 2\'; just write them normally\", \n'
                    '  \"chart_1\": {\n'
                    '    \"type\": \"histogram\" | \"bar\" | \"scatter\", \n'
                    '    \"x\": \"column_name_for_x\", \n'
                    '    \"y\": \"column_name_for_y_or_null\"\n'
                    "  },\n"
                    '  \"chart_2\": {\n'
                    '    \"type\": \"histogram\" | \"bar\" | \"scatter\", \n'
                    '    \"x\": \"column_name_for_x\", \n'
                    '    \"y\": \"column_name_for_y_or_null\"\n'
                    "  }\n"
                    "}\n"
                    "Use only columns that exist in the summary. But do not use the index or any unlisted columns in chart_1 or chart_2."
                    "- Do NOT choose exactly the same (type, x, y) for chart_1 and chart_2.\n"
                    "- If you cannot recommend a second chart, set chart_2 to null.\n"
                    "- round all values to 2 decimal places where applicable.\n"
                    "Do NOT add any extra keys or text outside the JSON."
                ),
            }
        ],
    )

    # Extract text blocks from the response
    raw = "".join(block.text for block in resp.content if block.type == "text")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if Claude's output isn't valid JSON
        parsed = {
            "text": raw,
            "chart": None,
        }

    # Ensure keys exist
    if "text" not in parsed:
        parsed["text"] = ""
    if "chart" not in parsed:
        parsed["chart"] = None

    return parsed

def build_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    n_rows, n_cols = df.shape

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_summary = {}
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        for col in desc.index:
            stats = desc.loc[col]
            numeric_summary[col] = {
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
                "min": float(stats["min"]),
                "max": float(stats["max"]),
            }

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_summary = {}
    for col in cat_cols[:5]:
        value_counts = df[col].value_counts().head(3)
        categorical_summary[col] = {str(k): int(v) for k, v in value_counts.items()}
    # Very simple chart suggestions
    chart_config = {
        "hist_column": numeric_cols[0] if len(numeric_cols) >= 1 else None,
        "scatter_x": numeric_cols[0] if len(numeric_cols) >= 2 else None,
        "scatter_y": numeric_cols[1] if len(numeric_cols) >= 2 else None,
    }

    summary = {
        "shape": {"rows": n_rows, "columns": n_cols},
        "numeric_columns": numeric_cols,
        "numeric_summary": numeric_summary,
        "categorical_columns": cat_cols,
        "categorical_summary": categorical_summary,
        "chart_config": chart_config,
    }
    return summary