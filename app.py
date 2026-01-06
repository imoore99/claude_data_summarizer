"""
app.py — CSV summarization orchestration using an LLM client.

Responsibilities:
- Load and clean CSV input.
- Orchestrate calls to a summarization client to produce text and matplotlib code.
- Persist summaries to disk.

Side effects:
- Reads environment variables (e.g., ANTHROPIC_API_KEY).
- Instantiates a module-level client by default; for tests, prefer injecting a client.

Public API:
- load_csv(path: str) -> list[dict]
- clean_rows(rows) -> list[dict]
- summarize_records(rows, client, ...) -> str
- save_summary(path: str, summary: str) -> None
- main(argv: Optional[list[str]]) -> int

Notes:
- Document the expected shape of returned objects (e.g., matplotlib_code string that defines a `fig`).
- Keep real client construction in `main()` and inject a fake client for unit tests.
"""

#import packages
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes  #for test datasets
from agent import summarize_with_claude, ask_followup_question, build_context_for_followup

## <------- Function Set STARTS Here ------> ##
def button_callback(label, href):
    button_html = f"""
    <a href="{href}" style="text-decoration: none; flex: 1 1 0; display: flex;">
        <button style="
            background-color: #005f85;
            color: white;
            padding: 5px 10px;
            text-align: center;
            text-decoration: none;
            width: 100%;
            font-size: 12px;
            cursor: pointer;
            border-radius: 0px;
            white-space: nowrap;
        ">
        {label}
        </button>
    </a>
    """
    return button_html

# Helper function at top of file
def extract_python_code(code_string: str) -> str:
    """Extract Python code from markdown code blocks."""
    if not code_string:
        return ""
    code = code_string.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()

def render_text_safely(text: str) -> None:
    """
    Render text safely without markdown interpretation issues.
    Handles code blocks separately from regular text.
    """
    if not text:
        return
    
    # Check if text contains code blocks
    if "```" in text:
        # Split on code blocks and render separately
        parts = re.split(r'(```[\w]*\n.*?```)', text, flags=re.DOTALL)
        
        for part in parts:
            if part.startswith("```"):
                # Extract language and code
                lines = part.split('\n', 1)
                if len(lines) > 1:
                    lang = lines[0].replace('```', '').strip() or 'python'
                    code = lines[1].replace('```', '').strip()
                    st.code(code, language=lang)
            elif part.strip():
                # Regular text - use text() to avoid markdown
                st.text(part.strip())
    else:
        # No code blocks - just render as text
        st.text(text)

def detect_visualization_intent(question: str) -> bool:
    """
    Detect if user wants a NEW visualization.
    
    Returns True if:
    - User explicitly asks to create/generate a chart
    - User asks for "another" or "different" visualization
    
    Returns False if:
    - User asks questions about existing charts
    - User requests code
    - User asks analytical questions without viz request
    """
    
    question_lower = question.lower()
    
    # Explicit chart creation requests
    creation_keywords = [
        "create", "generate", "make", "draw", "build",
        "show me a", "can you plot", "can you create",
        "another", "different", "new", "alternative"
    ]
    
    # Questions about existing content (don't need new viz)
    question_keywords = [
        "what is the", "what's the", "explain", "why",
        "how does", "tell me about", "describe"
    ]
    
    # Code requests
    code_keywords = ["code", "python", "script", "show me the code"]
    
    # Check for code request first
    if any(kw in question_lower for kw in code_keywords):
        return False  # Don't generate new chart
    
    # Check for questions about existing content
    if any(kw in question_lower for kw in question_keywords):
        # Exception: if they explicitly ask for a chart
        if any(word in question_lower for word in ["plot", "chart", "graph"]):
            # "What is another plot?" → True
            if any(word in question_lower for word in ["another", "different", "new"]):
                return True
        return False  # Just answer the question
    
    # Check for explicit creation requests
    has_creation_keyword = any(kw in question_lower for kw in creation_keywords)
    has_viz_keyword = any(word in question_lower for word in [
        "plot", "chart", "graph", "visualiz", "scatter", 
        "histogram", "heatmap", "box", "bar", "line"
    ])
    
    return has_creation_keyword and has_viz_keyword
## <------- Function Set ENDS Here ------> ##
    
# Page config
st.set_page_config(
    page_title="Data Summarizer Dashboard",
    page_icon="📈",
    layout="wide"
)

# After page config - set counters for later use
if "followup_history" not in st.session_state:
    st.session_state.followup_history = []

if "chart_history" not in st.session_state:
    st.session_state.chart_history = []

if "token_count" not in st.session_state:
    st.session_state.token_count = 0 

# Sticky header configuration
with stylable_container(
    key="sticky_header",
    css_styles="""
        {
            position: fixed;
            top: 2.875rem;
            background-color: #000000;
            z-index: 1000;
            width: 100%;
            border-bottom: 1px solid #ccc;
            padding: 5px 0px 50px 0px;
            box-sizing: border-box;
            display: flex;
            justify-content: flex-start;
            align-items: flex-start;
        }
    """,
):
    # Set Title
    st.title("DATA SUMMARIZER DASHBOARD")
    col_main1, col_main2 = st.columns([3,4])
    with col_main1:
        with stylable_container(
            key="nav_buttons",
            css_styles="""
                {
                    display: flex;
                    flex-direction: row;
                    justify-content: space-evenly;
                    align-items: center;
                    gap: 8px;
                    width: 100%;
                }
            """,
        ):
            nav_html = """
            <div style="display:flex; width:100%; justify-content:space-evenly; align-items:center; gap:8px;">
            """
            nav_html += button_callback("Back to the Top", "#dataset-summary-with-claude")
            nav_html += button_callback("Dataset Preview", "#dataset-preview")
            nav_html += button_callback("Dataset Description", "#dataset-description")
            nav_html += button_callback("Dataset Summary and Analysis", "#dataset-summary-and-analysis")
            nav_html += button_callback("Dataset Prompt", "#dataset-prompt")
            nav_html += "</div>"
            st.markdown(nav_html, unsafe_allow_html=True)


st.markdown("---")
## <------ Dataset Setup STARTS Here ------> ##
st.markdown("## Dataset Summary with Claude")
top_row = st.columns([3,3,3])
# Column 1
with top_row[0]:
    st.markdown("#### Function Summary")
    st.text("This app allows you to either use a predefined dataset or upload a CSV dataset and get a summary analysis using Claude. The analysis includes statistical summaries, data insights, and actionable recommendations. You can explore the dataset preview, get a detailed description, and receive a comprehensive summary and analysis to help you understand your data better.")
# Column 2
with top_row[1]:
    st.markdown("#### Choose a dataset")
    dataset_source = st.radio(
        "Data source",
        ["Upload CSV", "Sample dataset"],
        horizontal=True,
        label_visibility="collapsed",
    )

    df = None
    if dataset_source == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
    else:
        sample_name = st.selectbox(
            "Choose a sample dataset",
            ["Iris", "Wine", "Breast Cancer", "Diabetes"],
        )
        
        if sample_name == "Iris":
            data = load_iris(as_frame=True)
            df = data.frame
        elif sample_name == "Wine":
            data = load_wine(as_frame=True)
            df = data.frame
        elif sample_name == "Breast Cancer":
            data = load_breast_cancer(as_frame=True)
            df = data.frame
        elif sample_name == "Diabetes":
            data = load_diabetes(as_frame=True)
            df = data.frame

    if df is not None:
        st.session_state["df"] = df

    analyze_disabled = st.session_state.get("df") is None
    if st.button("Analyze with Claude", disabled=analyze_disabled):
        with st.spinner("Generating summary..."):
            df_current = st.session_state.get("df")
            claude_summary = summarize_with_claude(df_current)

        # Check for errors
        if claude_summary.get("error"):
            st.error(claude_summary.get("text"))
            # Don't save to session state if error
        else:

            st.session_state["claude_text"] = claude_summary.get("text")
            st.session_state["next_steps"] = claude_summary.get("next_steps")
            st.session_state['chart_1'] = claude_summary.get("chart_1")
            st.session_state['chart_2'] = claude_summary.get("chart_2")
            st.session_state["matplotlib_code"] = claude_summary.get("matplotlib_code")
# Column 3
with top_row[2]:
    st.markdown("#### Instructions")
    st.markdown(
        """
        1. Upload a CSV file or choose a sample dataset.
        2. Click "Analyze with Claude" to get an initial EDA-style summary.
        3. View the summary analysis provided by Claude.
        4. Use the chat interface below to ask for additional insights or visualizations.
        """
    )
## <------ Dataset Setup ENDS Here ------> ##

st.markdown("---")
## <------ Dataset Preview STARTS Here ------> ##
st.markdown("## Dataset Preview")
# Safe preview at the bottom
df_preview = st.session_state.get("df")
if df_preview is not None:
    st.dataframe(df_preview.head())
else:
    st.write("No data uploaded yet.")
## <------ Dataset Preview ENDS Here ------> ##

st.markdown("---")
## <------ Dataset Description STARTS Here ------> ##
st.markdown("## Dataset Description")
df_preview = st.session_state.get("df")
if df_preview is not None:
    st.dataframe(df_preview.describe())
else:
    st.write("No data uploaded yet.")
## <------ Dataset Description ENDS Here ------> ##

st.markdown("---")
## <------ Dataset Claude Summary STARTS Here ------> ##
st.markdown("## Dataset Summary and Analysis")
st.markdown("##### Claude's Analysis:")
claude_text = st.session_state.get("claude_text")
claude_next_steps = st.session_state.get("next_steps")
claude_chart_1 = st.session_state.get("chart_1")
claude_chart_2 = st.session_state.get("chart_2")
claude_code = st.session_state.get("matplotlib_code")
if claude_text:
    st.markdown("**Summary:**\n")
    render_text_safely(claude_text)
    st.markdown("**Next Steps:**\n")
    render_text_safely(claude_next_steps)
    st.markdown(f"**Chart 1:** {claude_chart_1}")
    st.markdown(f"**Chart 2:** {claude_chart_2}")
    
    # Fix code execution
    if claude_code:
        try:
            # Clean markdown code blocks
            clean_code = extract_python_code(claude_code)
            
            # Execute
            namespace = {
                'df': st.session_state.get("df"),
                'plt': plt,
                'np': np,
                'pd': pd
            }
            exec(clean_code, namespace)
            
            # Display figure
            if 'fig' in namespace:
                st.pyplot(namespace['fig'])
            else:
                st.error("No figure created by the code")
                
        except SyntaxError as e:
            st.error(f"⚠️ Code syntax error: {e}")
            with st.expander("Show problematic code"):
                st.code(claude_code, language="python")
                
        except Exception as e:
            st.error(f"⚠️ Execution error: {e}")
            with st.expander("Show code"):
                st.code(clean_code, language="python")
else:
    st.write("No summary available yet.")
## <------ Dataset Claude Summary STARTS Here ------> ##
    
st.markdown("---")
## <------ Claude Prompt Summary STARTS Here ------> ##
st.markdown("## Dataset Prompt")
# Add turn counter and token counter
turn_count = len(st.session_state.followup_history) // 2
token_count = st.session_state.token_count
st.caption(f"Conversation turns: {turn_count}/10 | Tokens: {token_count}/25000")
# turn counter and token counter statement tracking
if turn_count >= 10 or token_count >= 25000:
    st.error("⚠️ Conversation limit reached (10 turns or 25000 tokens). Click below to start fresh.")
    if st.button("Clear Chat History"):
        st.session_state.followup_history = []
        st.rerun()
elif turn_count >= 8 or token_count >= 20000:
    st.warning("⚠️ Conversation limit nearly reached (10 turns or 25000 tokens).")

if turn_count < 10 and token_count < 25000:
    st.text_input("Ask a follow-up question about the data:", key="followup_input")
    if st.button("Ask Claude"):
        followup = st.session_state.get("followup_input", "")
        if followup:
            df_current = st.session_state.get("df")
            df_context = build_context_for_followup(df_current)
            
            # USE THE DETECTION FUNCTION
            force_tool = detect_visualization_intent(followup)
            
            # Optional: Show user what you detected (debugging)
            # st.caption(f"Debug: Detected viz intent = {force_tool}")
            
            with st.spinner("Thinking..."):
                followup_result, token_count = ask_followup_question(
                    followup, 
                    st.session_state.followup_history,
                    df_context,
                    force_tool=force_tool  # Pass the decision
                )
                
            st.session_state.token_count += token_count
            # Store ONLY role and content in history (for API)
            st.session_state.followup_history.append({
                "role": "user",
                "content": followup
            })
            st.session_state.followup_history.append({
                "role": "assistant",
                "content": followup_result.get("text")
            })
            
            # Initialize chart history if needed
            if "chart_history" not in st.session_state:
                st.session_state.chart_history = []
            
            # Append new chart (order matters!)
            st.session_state.chart_history.append({
                "question": followup,
                "code": followup_result.get("matplotlib_code"),
                "chart_1": followup_result.get("chart_1"),
                "chart_2": followup_result.get("chart_2")
            })

        else:
            # No chart generated, append None to maintain alignment
            if "chart_history" not in st.session_state:
                st.session_state.chart_history = []
            st.session_state.chart_history.append(None)
            
            st.rerun()

# Display conversation history
if st.session_state.followup_history:
    st.markdown("### Conversation")
    
    # Initialize chart history if needed
    if "chart_history" not in st.session_state:
        st.session_state.chart_history = []
    
    chart_index = 0  # Track which chart we're on
    
    for i, msg in enumerate(st.session_state.followup_history):
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Claude:** {msg['content']}")
            
            # Check if there's a chart for this assistant response
            # Assistant messages are at odd indices (1, 3, 5, ...)
            assistant_response_num = (i - 1) // 2  # Which assistant response is this?
            
            if assistant_response_num < len(st.session_state.chart_history):
                chart = st.session_state.chart_history[assistant_response_num]
                # In your chart display section:
                if chart and chart.get("code"):
                    try:
                        from scipy.spatial import ConvexHull
                    except ImportError:
                        ConvexHull = None
                    
                    try:
                        from io import StringIO
                    except ImportError:
                        StringIO = None
                    
                    # Create namespace INSIDE the try block
                    namespace = {
                        'df': st.session_state.get("df"),
                        'plt': plt,
                        'np': np,
                        'pd': pd,
                    }
                    
                    # Add optional imports
                    if ConvexHull:
                        namespace['ConvexHull'] = ConvexHull
                    if StringIO:
                        namespace['StringIO'] = StringIO
                    
                    try:
                        # Execute matplotlib code
                        exec(chart["code"], namespace)
                        # Get the fig from namespace
                        if 'fig' in namespace:
                            st.pyplot(namespace['fig'])
                        else:
                            st.error("No 'fig' object created by the code")
                    except Exception as e:
                        st.error(f"Error rendering chart: {e}")
                        st.code(chart["code"], language="python")  # Show the problematic code