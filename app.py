import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from agent import build_dataset_summary, summarize_with_claude

# Page config
st.set_page_config(
    page_title="Data Summarizer Dashboard",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("DATA SUMMARIZER DASHBOARD")
#st.markdown("#### This app summarizes datasets using Anthropic's Claude model.")

st.markdown("---")

top_row = st.columns([3,3,3])
with top_row[0]:
    st.markdown("#### Dataset Summary with Claude")
    st.text("This app allows you to upload a CSV dataset and get a summary analysis using Claude.")
with top_row[1]:
    st.markdown("#### Upload your dataset (CSV) for summarization")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # store in session_state so we can show it later
        st.session_state["df"] = df
        
    # Button goes here:
    if st.button("Analyze with Claude"):
        with st.spinner("Generating summary..."):
            summary = build_dataset_summary(df)
            claude_summary = summarize_with_claude(summary)

        # store in session_state so we can show it later
        st.session_state["claude_text"] = claude_summary["text"]
        st.session_state["summary"] = summary
        st.session_state["claude_chart_1"] = claude_summary.get("chart_1")
        st.session_state["claude_chart_2"] = claude_summary.get("chart_2")

with top_row[2]:
    st.markdown("#### Instructions")
    st.markdown(
        """
        1. Upload a CSV file using the uploader above.
        2. Wait for Claude to analyze the dataset.
        3. View the summary analysis provided by Claude.
        """
    )

st.markdown("---")
st.markdown("### Data Visualizations")

chart_spec_1 = st.session_state.get("claude_chart_1")
chart_spec_2 = st.session_state.get("claude_chart_2")
df_preview = st.session_state.get("df")

if chart_spec_1 and df_preview is not None:
    ctype_1 = chart_spec_1.get("type")
    x1 = chart_spec_1.get("x")
    y1 = chart_spec_1.get("y")
    ctype_2 = chart_spec_2.get("type") if chart_spec_2 else None
    x2 = chart_spec_2.get("x") if chart_spec_2 else None
    y2 = chart_spec_2.get("y") if chart_spec_2 else None

    cols = st.columns(2)
    with cols[0]:
        if ctype_1 == "histogram" and x1 in df_preview.columns:
            st.markdown(f"#### Claude-chosen histogram of {x1}")
            # Bin values
            binned = pd.cut(df_preview[x1], bins=5)
            counts = binned.value_counts().sort_index()
            # Convert interval index to strings for Altair
            hist_df = counts.reset_index()
            hist_df.columns = ["bin", "count"]
            hist_df["bin"] = hist_df["bin"].astype(str)

            chart = (
                alt.Chart(hist_df)
                .mark_bar()
                .encode(
                    x=alt.X("bin:N", title=f"{x1} (binned)"),
                    y=alt.Y("count:Q", title="Count"),
                )
            )
            st.altair_chart(chart, use_container_width=True)
        elif ctype_1 == "bar" and x1 in df_preview.columns:
            st.markdown(f"#### Claude-chosen bar chart of {x1}")
            st.bar_chart(df_preview[x1].value_counts())
        elif ctype_1 == "scatter" and x1 in df_preview.columns and y1 in df_preview.columns:
            st.markdown(f"#### Claude-chosen scatter: {x1} vs {y1}")
            st.scatter_chart(df_preview[[x1, y1]])
        else:
            st.write("Claude suggested a chart that can't be drawn with this data.")
    with cols[1]:
        if ctype_2 == "histogram" and x2 in df_preview.columns:
            st.markdown(f"#### Claude-chosen histogram of {x2}")
            st.bar_chart(df_preview[x2])
        elif ctype_2 == "bar" and x2 in df_preview.columns:
            st.markdown(f"#### Claude-chosen bar chart of {x2}")
            st.bar_chart(df_preview[x2].value_counts())
        elif ctype_2 == "scatter" and x2 in df_preview.columns and y2 in df_preview.columns:
            st.markdown(f"#### Claude-chosen scatter: {x2} vs {y2}")
            st.scatter_chart(df_preview[[x2, y2]])
        else:
            st.write("Claude suggested a chart that can't be drawn with this data.")
else:
    st.write("No Claude-chosen chart yet. Upload a file and click 'Analyze with Claude'.")

st.markdown("---")
st.markdown("### Data Summary and Claude Analysis")

st.markdown("##### Dataset preview:")
# Safe preview at the bottom
df_preview = st.session_state.get("df")
if df_preview is not None:
    st.dataframe(df_preview.head())
else:
    st.write("No data uploaded yet.")

st.markdown("##### Claude's Analysis:")
claude_text = st.session_state.get("claude_text")
if claude_text:
    st.markdown(claude_text)
else:
    st.write("No summary available yet.")



