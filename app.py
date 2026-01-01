import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import pandas as pd
import numpy as np
import altair as alt
import json

from agent import summarize_with_claude

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
    
# Page config
st.set_page_config(
    page_title="Data Summarizer Dashboard",
    page_icon="📈",
    layout="wide"
)

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
    # Title
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

st.markdown("## Dataset Summary with Claude")
top_row = st.columns([3,3,3])
with top_row[0]:
    st.markdown("#### Function Summary")
    st.text("This app allows you to upload a CSV dataset and get a summary analysis using Claude.")
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
        from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes

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

        st.session_state["claude_text"] = claude_summary.get("text")
        st.session_state["next_steps"] = claude_summary.get("next_steps")
        st.session_state['chart_1'] = claude_summary.get("chart_1")
        st.session_state['chart_2'] = claude_summary.get("chart_2")
        st.session_state["matplotlib_code"] = claude_summary.get("matplotlib_code")

with top_row[2]:
    st.markdown("#### Instructions")
    st.markdown(
        """
        1. Upload a CSV file or choose a sample dataset.
        2. Click "Analyze with Claude" to get an initial EDA-style summary.
        3. View the summary analysis provided by Claude.
        """
    )

st.markdown("---")
st.markdown("## Dataset Preview")
# Safe preview at the bottom
df_preview = st.session_state.get("df")
if df_preview is not None:
    st.dataframe(df_preview.head())
else:
    st.write("No data uploaded yet.")

st.markdown("---")
st.markdown("## Dataset Description")
df_preview = st.session_state.get("df")
if df_preview is not None:
    st.dataframe(df_preview.describe())
else:
    st.write("No data uploaded yet.")


st.markdown("---")
st.markdown("## Dataset Summary and Analysis")

st.markdown("##### Claude's Analysis:")
claude_text = st.session_state.get("claude_text")
claude_next_steps = st.session_state.get("next_steps")
claude_chart_1 = st.session_state.get("chart_1")
claude_chart_2 = st.session_state.get("chart_2")
claude_code = st.session_state.get("matplotlib_code")
if claude_text:
    st.markdown(f"**Summary:**\n {claude_text}")
    st.markdown(f"**Next Steps:**\n {claude_next_steps}")
    st.markdown(f"**Chart 1:** {claude_chart_1}")
    st.markdown(f"**Chart 2:** {claude_chart_2}")
    
    # Execute the code to generate the figure
    exec(claude_code)

    # Pass the fig object to streamlit
    st.pyplot(fig)
else:
    st.write("No summary available yet.")

st.markdown("---")
st.markdown("## Dataset Prompt")
st.text_input("Ask a follow-up question about the data:", key="followup_input")
if st.button("Ask Claude"):
    followup = st.session_state.get("followup_input", "")
    if followup:
        claude_response = st.session_state.get("claude_response", {})
        chart_1 = claude_response.get("chart_1")
        chart_2 = claude_response.get("chart_2")
        summary = st.session_state.get("summary", "")
        
        followup_result = ask_followup_question(followup, summary, chart_1, chart_2)
        st.session_state["claude_text"] = followup_result.get("text")
        st.session_state["claude_chart_1"] = followup_result.get("chart_1")
        st.session_state["claude_chart_2"] = followup_result.get("chart_2")
        st.session_state["followup_history"].append({"role": "user", "text": followup})
        st.session_state["followup_history"].append({"role": "assistant", "text": followup_result.get("text")})
        st.rerun()
    