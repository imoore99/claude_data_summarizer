# claude_data_summarizer

An AI-powered exploratory data analysis tool that provides instant insights and conversational data exploration using Claude Sonnet 4.

---

## What It Does

Upload any CSV dataset and get:
- **Automated insights** - AI-generated summary of key patterns and trends
- **Smart visualizations** - Matplotlib charts created on-demand based on your data
- **Conversational exploration** - Ask follow-up questions like "What's the average for class 0?" or "Show me a correlation heatmap"
- **Context-aware responses** - AI remembers your conversation and dataset structure

---

## Features

- **Direct Claude API Integration** - Built from scratch without frameworks to understand fundamentals
- **10-Turn Conversations** - Intelligent conversation limits with token tracking
- **Dynamic Chart Generation** - AI decides when visualizations are needed
- **Smart Intent Detection** - Distinguishes between questions, code requests, and visualization needs
- **Session Management** - Maintains conversation context across interactions
- **Sample Datasets** - Includes Iris, Wine, Breast Cancer, and Diabetes datasets

---

## Quick Start

### Prerequisites
- Python 3.8+
- Anthropic API key ([sign up here](https://console.anthropic.com/))

### Installation
```bash
# Clone the repository
git clone https://github.com/imoore99/claude_data_summarizer.git
cd claude-data-summarizer

# Install dependencies
pip install -r requirements.txt

# Set up your API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# Run the app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## How to Use

1. **Choose Your Data**
   - Upload a CSV file, OR
   - Select a sample dataset (Iris, Wine, etc.)

2. **Get Initial Analysis**
   - Click "Analyze with Claude"
   - Review AI-generated insights and chart recommendations

3. **Ask Follow-Up Questions**
```
   "What's the average alcohol content for class 0?"
   "Create a scatter plot of petal length vs width"
   "Show me a heatmap of correlations"
   "What patterns do you see in the South region?"
```

4. **View Visualizations**
   - Charts are generated automatically based on your questions
   - Code is executed in real-time with matplotlib

---

## Architecture
```
┌─────────────────────────────────────────────┐
│           Streamlit Frontend                │
│  (app.py - UI & conversation management)    │
└──────────────────┬──────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────┐
│         Agent Layer (agent.py)              │
│  - API calls to Claude                      │
│  - Tool orchestration                       │
│  - Context building                         │
└──────────────────┬──────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────┐
│      Anthropic Claude Sonnet 4 API          │
│  - Natural language understanding           │
│  - Tool use (generate_summary)              │
│  - Code generation                          │
└──────────────────┬──────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────┐
│         Generated Output                    │
│  - Text insights                            │
│  - Matplotlib code                          │
│  - Visualizations                           │
└─────────────────────────────────────────────┘
```

**Key Design Patterns:**
- **Intent Detection** - Keyword-based logic to determine when charts are needed
- **Conversation State** - Maintains history without sending full dataset repeatedly
- **Token Optimization** - Sends summary statistics instead of raw data
- **Tool Orchestration** - Conditional forcing of tools based on user intent

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| AI Model | Anthropic Claude Sonnet 4 |
| API Integration | Direct API calls (no frameworks) |
| Frontend | Streamlit |
| Backend | Python 3.8+ |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Deployment | Railway |

---

## What I Learned

This was my first AI agent project. Key learnings:

### **Technical Skills**
- **API Integration** - Understanding stateless LLMs and conversation management
- **Prompt Engineering** - Crafting system prompts to guide AI behavior
- **Tool Use** - Implementing function calling with conditional logic
- **Token Economics** - Optimizing context to reduce API costs
- **Error Handling** - Graceful degradation for API failures

### **Architectural Decisions**
- **Separation of Concerns** - Business logic (app.py) vs. API calls (agent.py)
- **Intent Detection** - When to use code vs. AI for decision-making
- **State Management** - What to store in session vs. what to send to API
- **Context Building** - Balancing information completeness with token efficiency

### **Key Insights**
1. LLMs are stateless - YOU manage conversation history
2. Token optimization is crucial - don't send unnecessary data
3. Prompt engineering > complex code for guiding behavior
4. Direct API integration teaches fundamentals better than frameworks

---

## What I'd Do Differently Next Time

- **Use LangChain** - For production-grade agent orchestration
- **Better Intent Detection** - ML-based classification vs. keyword matching
- **Enhanced Error Handling** - More granular error messages and retry logic
- **Streaming Responses** - Show AI "thinking" in real-time
- **Export Functionality** - Download analysis as PDF/HTML reports
- **Persistent History** - Save conversations across sessions
- **Chart Refinement** - Allow users to request modifications to generated charts

---

## Known Limitations

- Works best with clean, numeric datasets
- May struggle with very large files (>1000 rows)
- Text-heavy or categorical-only data produces limited visualizations
- Occasional inconsistency in chart generation (AI decision-making)
- No data validation on CSV uploads

**These are learning opportunities for future projects!**

---

## Roadmap

### **Project 2 (Feb-April 2025): AI Portfolio Optimizer**
Next project will demonstrate progression:
- LangChain agent framework
- Financial domain (CAPM, VaR, Efficient Frontier)
- Plotly interactive visualizations
- Risk-based recommendations

---

## Testing

**Recommended test datasets:**
- Iris (multiclass classification)
- Wine (quality prediction)
- Sales data (business analytics)
- Custom CSV with numeric columns

**Known edge cases:**
- Very large datasets (>5000 rows) may be slow
- Datasets with many text columns may produce limited insights

---

### LIVE PROJECT:
- View Full Analysis & Visualizations → https://claudedatasummarizer-production.up.railway.app/
- Explore the complete project to test EDA and conversational data analysis generation

### CONTACT:

Ian Moore - Business Intelligence, Credit Risk and Financial Analytics Leader

📧 EMAIL: ian.moore@hey.com

💼 LinkedIn: https://www.linkedin.com/in/ian-moore-analytics/

🌐 Portfolio: https://www.ianmooreanalytics.com

---

**⚠️ Note:** This is a learning project demonstrating AI API integration fundamentals. For production use, additional security, testing, and error handling would be required. Please experiment with caution.
```

