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


description = (
    "You are an experienced business intelligence analyst."
    "Generate a structured summary of dataset analysis with visualization recommendation."
    "Provide a natural language summary of key findings and insights, plus two recommended charts the user can create."
    "Provide a recommendation on the next step in the analysis for the user."
    "Focus on actionable insights and clear visualizations."
    "Return the matplotlib code to generate the charts."
    "Return the code as a string in a code block."
)

matplotlib_description = (
    "Complete executable matplotlib code that creates both charts in a figure with two subplots. "
    "The code should create a figure object named 'fig' and return it at the end. "
    "Assumes df is already loaded. Example: fig, axes = plt.subplots(1, 2, figsize=(12, 5))"
)

summary_tool = {
    "name": "generate_summary",
    "description": description,
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Natural language summary of key findings and insights"
            },
            "next_steps": {
                "type": "string",
                "description": "Recommendation on the next step in the analysis"
            },
            "chart_1": {
                "type": "object",
                "description": "First recommended chart",
                "properties": {
                    "type": {"type": "string", "description": "Chart type (histogram, scatter, box, bar)"},
                    "x": {"type": "string", "description": "X-axis column name"},
                    "y": {"type": "string", "description": "Y-axis column name (null for histograms)"}
                },
                "required": ["type", "x"]
            },
            "chart_2": {
                "type": "object",
                "description": "Second recommended chart",
                "properties": {
                    "type": {"type": "string"},
                    "x": {"type": "string"},
                    "y": {"type": "string"}
                    }
                },
            "matplotlib_code": {
                "type": "string",
                "description": matplotlib_description
        }
        },
        "required": ["text", "next_steps", "chart_1", "chart_2", "matplotlib_code"]
    }
}

def summarize_with_claude(summary_text) -> str:

    # Call Claude with the dataset summary and return a short analysis.
    # Include chart suggestions if available.
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        tools=[summary_tool],
        messages=[{
            "role": "user",
            "content": f"Analyze this dataset summary and provide insights:\n\n{summary_text}"
        }],
    )
    
    # Extract the tool use input (already a dict)
    for block in response.content:
        if block.type == 'tool_use' and block.name == 'generate_summary':
            return block.input  # Return dict directly, not json.dumps()
    
    return None