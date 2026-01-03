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

def ask_followup_question(question: str, conversation_history: list, df_context: str, force_tool: bool = False) -> Dict[str, Any]:
    """Handle follow-up questions with conversation context."""
    
    # Build messages - ONLY include role and content
    messages = []
    
    # Add system context
    messages.append({
        "role": "user",
        "content": f"""{df_context}

            You have access to a tool called 'generate_summary' that can create visualizations.
            When the user asks for charts or plots, you MUST use the generate_summary tool to return executable matplotlib code.

            Please help me analyze this data."""
    })
    messages.append({
        "role": "assistant", 
        "content": "I understand the dataset and will use the generate_summary tool when you need visualizations. What would you like to know?"
    })
    
    # Add conversation history
    for msg in conversation_history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # Add current question
    messages.append({
        "role": "user",
        "content": question
    })
    
    # DECIDE TOOL CHOICE BASED ON force_tool PARAMETER
    if force_tool:
        tool_choice = {"type": "tool", "name": "generate_summary"}  # FORCE the tool
    else:
        tool_choice = {"type": "auto"}  # Let Claude decide
    
    # Call Claude
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=3000,
        tools=[summary_tool],
        tool_choice=tool_choice,  # <-- USE THE CONDITIONAL CHOICE
        messages=messages
    )
    
    # Parse response
    result = {
        "text": "",
        "chart_1": None,
        "chart_2": None,
        "matplotlib_code": None
    }
    
    for block in response.content:
        if block.type == 'text':
            result["text"] += block.text
        elif block.type == 'tool_use' and block.name == 'generate_summary':
            result["text"] = block.input.get("text", "")
            result["chart_1"] = block.input.get("chart_1")
            result["chart_2"] = block.input.get("chart_2")
            result["matplotlib_code"] = block.input.get("matplotlib_code")
    
    return result

def build_context_for_followup(df):
    """Build intelligent context that works for any dataset."""
    context_parts = []
    
    # Basic info
    context_parts.append(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
    context_parts.append(f"Columns: {', '.join(df.columns)}")
    
    # Data types
    context_parts.append(f"\nData types:\n{df.dtypes.to_string()}")
    
    # Overall summary for numeric columns (compact)
    context_parts.append("\nNumeric column summary:")
    context_parts.append(df.describe().to_string())
    
    # Detect potential grouping columns (low cardinality)
    categorical_cols = []
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() < 20:
            categorical_cols.append(col)
    
    if categorical_cols:
        context_parts.append(f"\nPotential grouping columns: {', '.join(categorical_cols)}")
        # Show value counts for categorical columns
        for col in categorical_cols[:3]:  # Limit to first 3 to save tokens
            value_counts = df[col].value_counts()
            context_parts.append(f"\n{col} distribution:\n{value_counts.to_string()}")
    
    context_parts.append("\nIMPORTANT: Variable 'df' contains the full dataset and is available for all plotting operations.")
    
    return "\n".join(context_parts)