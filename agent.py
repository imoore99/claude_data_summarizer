#import os
from typing import Dict, Any
# import json

# import pandas as pd
from anthropic import Anthropic
import logging


from dotenv import load_dotenv

# Load .env so ANTHROPIC_API_KEY is available
load_dotenv()

# Create Anthropic client (reads ANTHROPIC_API_KEY from environment)
client = Anthropic()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    
    try:
        
        logger.info(f"API call started")
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
        logger.info(f"API response successful")
        
        # Extract the tool use input (already a dict)
        for block in response.content:
            if block.type == 'tool_use' and block.name == 'generate_summary':
                return block.input  # Return dict directly, not json.dumps()

        # If no tool use found, return error
        logger.warning("No tool use found in response")
        return {
            "text": "⚠️ Unable to generate summary. Please try again.",
            "next_steps": "",
            "chart_1": None,
            "chart_2": None,
            "matplotlib_code": None,
            "error": True
        }, 0
    
    except anthropic.APIConnectionError as e:
        logger.error(f"Connection error: {str(e)}")
        return {
            "text": "⚠️ Connection error: Unable to reach the AI service. Please check your internet connection and try again.",
            "next_steps": "",
            "chart_1": None,
            "chart_2": None,
            "matplotlib_code": None,
            "error": True
        }, 0
    
    except anthropic.RateLimitError as e:
        logger.warning(f"Rate limit hit: {str(e)}")
        return {
            "text": "⚠️ Rate limit reached: Too many requests. Please wait a moment and try again.",
            "next_steps": "",
            "chart_1": None,
            "chart_2": None,
            "matplotlib_code": None,
            "error": True
        }, 0
    
    except anthropic.APIStatusError as e:
        logger.error(f"API status error {e.status_code}: {str(e)}")
        if e.status_code == 400:
            error_msg = "⚠️ Invalid request: There was an issue analyzing this dataset. Please try a different dataset."
        elif e.status_code == 401:
            error_msg = "⚠️ Authentication error: API key is invalid. Please check your configuration."
        elif e.status_code >= 500:
            error_msg = "⚠️ Service temporarily unavailable: The AI service is experiencing issues. Please try again in a few moments."
        else:
            error_msg = f"⚠️ Error {e.status_code}: An unexpected error occurred. Please try again."
        
        return {
            "text": error_msg,
            "next_steps": "",
            "chart_1": None,
            "chart_2": None,
            "matplotlib_code": None,
            "error": True
        }, 0
    
    except Exception as e:
        logger.error(f"Unexpected error in summarize_with_claude: {str(e)}")
        return {
            "text": f"⚠️ Unexpected error: {str(e)}. Please try again or contact support if the issue persists.",
            "next_steps": "",
            "chart_1": None,
            "chart_2": None,
            "matplotlib_code": None,
            "error": True
        }, 0

    

def ask_followup_question(question: str, conversation_history: list, df_context: str, force_tool: bool = False) -> Dict[str, Any]:
    """Handle follow-up questions with conversation context."""
    try:
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
        
        logger.info(f"API call started - Question: {question[:50]}")
        # Call Claude
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            tools=[summary_tool],
            tool_choice=tool_choice,  # <-- USE THE CONDITIONAL CHOICE
            messages=messages
        )
        logger.info("API call successful")
        
        # Parse response
        result = {
            "text": "",
            "chart_1": None,
            "chart_2": None,
            "matplotlib_code": None,
            "error": False
        }
        
        for block in response.content:
            if block.type == 'text':
                result["text"] += block.text
            elif block.type == 'tool_use' and block.name == 'generate_summary':
                result["text"] = block.input.get("text", "")
                result["chart_1"] = block.input.get("chart_1")
                result["chart_2"] = block.input.get("chart_2")
                result["matplotlib_code"] = block.input.get("matplotlib_code")
        
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        token_count = input_tokens + output_tokens
        
        return result, token_count
    
    except anthropic.APIConnectionError as e:
        logger.error(f"Connection error: {str(e)}")
        return {
            "text": "⚠️ Connection error: Unable to reach the AI service. Please check your internet connection and try again.",
            "chart_1": None,
            "chart_2": None,
            "matplotlib_code": None,
            "error": True
        }, 0
    
    except anthropic.RateLimitError as e:
        logger.warning(f"Rate limit hit: {str(e)}")
        return {
            "text": "⚠️ Rate limit reached: Too many requests. Please wait a moment and try again.",
            "chart_1": None,
            "chart_2": None,
            "matplotlib_code": None,
            "error": True
        }, 0
    
    except anthropic.APIStatusError as e:
        logger.error(f"API status error {e.status_code}: {str(e)}")
        if e.status_code == 400:
            error_msg = "⚠️ Invalid request: There was an issue with the request format. Please try rephrasing your question."
        elif e.status_code == 401:
            error_msg = "⚠️ Authentication error: API key is invalid. Please contact support."
        elif e.status_code >= 500:
            error_msg = "⚠️ Service temporarily unavailable: The AI service is experiencing issues. Please try again in a few moments."
        else:
            error_msg = f"⚠️ Error {e.status_code}: An unexpected error occurred. Please try again."
        
        return {
            "text": error_msg,
            "chart_1": None,
            "chart_2": None,
            "matplotlib_code": None,
            "error": True
        }, 0
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            "text": f"⚠️ Unexpected error: {str(e)}. Please try again or contact support if the issue persists.",
            "chart_1": None,
            "chart_2": None,
            "matplotlib_code": None,
            "error": True
        }, 0

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
    
    # CRITICAL: Tell Claude the dataframe is available
    context_parts.append("\n" + "="*50)
    context_parts.append("IMPORTANT FOR CODE GENERATION:")
    context_parts.append("="*50)
    context_parts.append("A pandas DataFrame named 'df' is available in the execution environment.")
    context_parts.append("You can use standard pandas operations for aggregations:")
    context_parts.append("  - df.groupby('column').agg({'other_col': 'mean'})")
    context_parts.append("  - df[df['column'] > value]")
    context_parts.append("  - df.pivot_table(...)")
    context_parts.append("All necessary imports (pandas, numpy, matplotlib) are available.")
    context_parts.append("Generate code that uses 'df' directly - it contains the full dataset.")
    
    
    return "\n".join(context_parts)