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

### ⚙️ Features

| Component                    | Description                                                                                 |
|-----------------------------|---------------------------------------------------------------------------------------------|
| **FastAPI Web UI**          | Friendly dashboard for synthetic or uploaded datasets                                      |
| **MCP Tool Orchestration**  | Each process step is exposed as a callable MCP tool                                        |
| **Anomaly Detection**       | Mahalanobis Distance-based outlier detection                                               |
| **Visual Output**           | Saved scatter plot of inliers vs. outliers                                                 |
| **Local LLM Summarization** | Insights generated using `llama3.2:1b` via Ollama                                           |
| **Vector Store Logging**    | Summaries are stored in ChromaDB for persistent memory                                     |
| **Agentic Planning Tool**   | A dedicated LLM tool (`autonomous_plan`) determines next steps based on dataset context    |
| **Agentic Flow**            | LLM + memory + tool use + automatic reasoning + context awareness                          |

---

### 🧪 Tools Defined (via MCP)

| Tool Name                     | Description                                                                                      | LLM Used |
|------------------------------|--------------------------------------------------------------------------------------------------|----------|
| `generate_data`              | Create synthetic tabular data (Gaussian + categorical)                                           | ❌        |
| `analyze_outliers`           | Label rows using Mahalanobis distance                                                           | ❌        |
| `plot_results`               | Save a plot visualizing inliers vs outliers                                                     | ❌        |
| `summarize_results`          | Interpret and explain outlier distribution using `llama3.2:1b`                                  | ✅        |
| `summarize_data_stats`       | Describe dataset trends using `llama3.2:1b`                                                     | ✅        |
| `log_results_to_vector_store`| Store summaries to ChromaDB for future reference                                                 | ❌        |
| `search_logs`                | Retrieve relevant past sessions using vector search (optional LLM use)                         | ⚠️        |
| `autonomous_plan`            | Run the full pipeline, use LLM to recommend next actions automatically                          | ✅        |

---

### 🤖 Agentic Capabilities

- **Autonomy**: LLM-guided execution path selection with `autonomous_plan`
- **Tool Use**: Dynamically invokes registered MCP tools via LLM inference
- **Reasoning**: Generates technical insights from dataset conditions and outlier analysis
- **Memory**: Persists and recalls knowledge using ChromaDB vector search
- **LLM**: Powered by Ollama with `llama3.2:1b` (temperature = 0.1, deterministic)


---

## 🚀 Getting Started

### 1. Clone and Set Up
```bash
git clone https://github.com/MadMando/autonomous-analyst.git
cd autonomous-analyst
conda create -n mcp-agentic python=3.11 -y
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

## 🙌 Contributions & Acknowledgements

This project wouldn't be possible without the incredible work of the open-source community. Special thanks to:

| Tool / Library              | Purpose                                         | Repository |
|----------------------------|-------------------------------------------------|------------|
| 🧠 **Model Context Protocol (MCP)** | Agentic tool orchestration & execution        | [modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk) |
| 💬 **Ollama**              | Local LLM inference engine (`llama3.2:1b`)      | [ollama/ollama](https://github.com/ollama/ollama) |
| 🔍 **ChromaDB**            | Vector database for logging and retrieval      | [chroma-core/chroma](https://github.com/chroma-core/chroma) |
| 🌐 **FastAPI**             | Interactive, fast web interface                | [tiangolo/fastapi](https://github.com/tiangolo/fastapi) |
| ⚡ **Uvicorn**             | ASGI server powering the FastAPI backend       | [encode/uvicorn](https://github.com/encode/uvicorn) |
| 📊 **pandas**              | Data manipulation and preprocessing            | [pandas-dev/pandas](https://github.com/pandas-dev/pandas) |
| 📈 **matplotlib**          | Data visualization (outlier plots)             | [matplotlib/matplotlib](https://github.com/matplotlib/matplotlib) |
| 🤖 **scikit-learn**        | Outlier detection and machine learning         | [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) |

> 💡 If you use this project, please consider starring or contributing to the upstream tools that make it possible.

This repo was created with the assistance of a local rag-llm using llama3.2:1b
