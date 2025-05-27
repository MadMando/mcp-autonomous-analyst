<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python Badge"/>
  <img src="https://img.shields.io/badge/MCP-Model_Context_Protocol-purple" alt="MCP Badge"/>
  <img src="https://img.shields.io/badge/Ollama-LLM-green" alt="Ollama Badge"/>
  <img src="https://img.shields.io/badge/ChromaDB-VectorDB-orange" alt="ChromaDB Badge"/>
  <img src="https://img.shields.io/badge/FastAPI-Web_UI-teal" alt="FastAPI Badge"/>
  <img src="https://img.shields.io/badge/Uvicorn-ASGI_Server-black" alt="Uvicorn Badge"/>
</p>

# Autonomous Analyst

## 🧠 Overview
Autonomous Analyst is a local, agentic AI pipeline that:
- Analyzes tabular data
- Detects anomalies with Mahalanobis distance
- Uses a local LLM (llama3.2:1b via Ollama) to generate interpretive summaries
- Logs results to ChromaDB for semantic recall
- Is fully orchestrated via the Model Context Protocol (MCP)

---

## ⚙️ Features
| Component                       | Description                                                                                  |
|--------------------------------|----------------------------------------------------------------------------------------------|
| `FastAPI` Web UI               | Friendly dashboard for synthetic or uploaded datasets                                        |
| MCP Tool Orchestration         | Each process step is exposed as a callable MCP tool                                          |
| Anomaly Detection              | Mahalanobis Distance-based outlier detection                                                 |
| Visual Output                  | Saved scatter plot of inliers vs. outliers                                                   |
| Local LLM Summarization       | Insights generated using `llama3.2:1b` via Ollama                                             |
| Vector Store Logging           | Logs summaries into ChromaDB for memory                                                      |
| Agentic Flow                   | LLM + memory + tool use + context awareness                                                  |

---

## 🧪 Tools Defined (via MCP)
| Tool Name                     | Description                                                                                    | LLM Used |
|------------------------------|------------------------------------------------------------------------------------------------|----------|
| `generate_data`              | Create synthetic tabular data (Gaussian + categorical)                                         | ❌        |
| `analyze_outliers`           | Label rows with Mahalanobis distance scoring                                                  | ❌        |
| `plot_results`               | Generate scatter plot visualization of anomaly classification                                | ❌        |
| `summarize_results`          | Use `llama3.2:1b` to interpret and explain the outlier findings                               | ✅        |
| `summarize_data_stats`       | Use `llama3.2:1b` to describe dataset stats in plain English                                 | ✅        |
| `log_results_to_vector_store`| Store LLM output summaries into ChromaDB                                                      | ❌        |
| `search_logs`                | Query past logs for similarity (uses ChromaDB, LLM optional)                                  | ⚠️        |

---

## 🤖 Agentic Capabilities
- **Autonomy:** Chooses workflow steps from data input to summary
- **Tool Use:** Tasks executed via standard MCP tool interface
- **Reasoning:** LLM interprets data conditions and suggests actions
- **Memory:** Uses ChromaDB for contextual continuity across sessions
- **LLM:** `llama3.2:1b` via Ollama with temperature=0.1

---

## 🚀 Getting Started

### 1. Clone and Set Up
```bash
git clone https://github.com/MadMando/autonomous-analyst.git
cd autonomous-analyst
conda create -n mcp-agentic python=3.10 -y
conda activate mcp-agentic
pip install uv
uv pip install -r requirements.txt
```

### 2. Start the MCP Server
```bash
mcp run server.py --transport streamable-http
```

### 3. Start the Web Dashboard
```bash
uvicorn web:app --reload
```
Then visit: [http://localhost:8000](http://localhost:8000)

---

## 🌐 Dashboard Flow

- **Step 1:** Upload your own dataset or click `Generate Synthetic Data`
- **Step 2:** The system runs anomaly detection on `feature_1` vs `feature_2`
- **Step 3:** Visual plot of outliers is generated
- **Step 4:** Summaries are created via LLM
- **Step 5:** Results are optionally logged to vector store for recall

---

## 📁 Project Layout
```
📦 autonomous-analyst/
├── server.py                  # MCP server
├── web.py                     # FastAPI + MCP client (frontend logic)
├── tools/
│   ├── synthetic_data.py
│   ├── outlier_detection.py
│   ├── plotter.py
│   ├── summarizer.py
│   ├── vector_store.py
├── static/                   # Saved plot
├── data/                     # Uploaded or generated dataset
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📚 Tech Stack
- **MCP SDK:** [`mcp`](https://github.com/modelcontextprotocol/python-sdk)
- **LLM Inference:** [Ollama](https://ollama.com/) running `llama3.2:1b`
- **UI Server:** FastAPI + Uvicorn
- **Memory:** ChromaDB vector database
- **Data:** `pandas`, `matplotlib`, `scikit-learn`

---

## ✅ .gitignore Additions
```
__pycache__/
*.pyc
*.pkl
.env
static/
data/
```

---
