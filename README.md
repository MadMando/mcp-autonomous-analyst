# 🧠 Autonomous Analyst

A local, agentic AI pipeline that analyzes tabular data, detects anomalies, summarizes insights using a local LLM (`llama3.2:1b`), and logs results to a vector store (ChromaDB) for future recall — all orchestrated via the Model Context Protocol (MCP).

---

## 📦 Features

| Module                     | Description                                                                                  |
|----------------------------|----------------------------------------------------------------------------------------------|
| **FastAPI Dashboard**      | Clean, single-page web UI for uploading data or generating synthetic samples.               |
| **MCP Tool Orchestration**| Each action is encapsulated in a tool callable via the MCP protocol.                        |
| **Outlier Detection**      | Mahalanobis Distance-based anomaly detection.                                                |
| **Visual Output**          | Scatter plot showing inliers vs. outliers.                                                   |
| **LLM-Powered Insights**   | Local Ollama LLM (`llama3.2:1b`) generates natural language summaries.                        |
| **Vector Logging**         | Summaries are logged to ChromaDB for future searchability.                                   |
| **Agentic Behavior**       | Agent autonomously chooses the flow, analyzes data, and contextualizes findings.             |

---

## 🛠️ Tools Available

| Tool Name                     | Description                                                                                      | LLM |
|------------------------------|--------------------------------------------------------------------------------------------------|-----|
| `generate_data`              | Synthesizes realistic datasets using normal and categorical sampling.                           | ❌  |
| `analyze_outliers`           | Flags anomalous rows using Mahalanobis distance.                                                 | ❌  |
| `plot_results`               | Generates a Matplotlib-based scatter plot of inliers vs. outliers.                               | ❌  |
| `summarize_results`          | Uses `llama3.2:1b` via Ollama to explain outliers and suggest next steps.                        | ✅  |
| `summarize_data_stats`       | Uses `llama3.2:1b` to interpret statistical patterns in natural language.                        | ✅  |
| `log_results_to_vector_store`| Persists summaries to ChromaDB for future reference.                                             | ❌  |
| `search_logs`                | Retrieves relevant sessions using semantic similarity (optional LLM support).                    | ⚠️  |

---

## 🧠 Agentic Workflow

This system simulates autonomous behavior through:

- 🤖 **Tool Use**: Each action is a callable MCP tool.
- 📥 **Perception**: Accepts user-uploaded data or generates its own.
- 🧮 **Reasoning**: Interprets results using a local LLM.
- 🧠 **Memory**: Logs sessions and enables semantic search over them.

**LLM Used**: `llama3.2:1b` via [Ollama](https://ollama.com/)  
**Inference Mode**: Deterministic, temperature = `0.1`

---

## 🚀 Quickstart

### 1. Clone and Set Up

```bash
git clone https://github.com/yourname/autonomous-analyst.git
cd autonomous-analyst
conda create -n mcp-agentic python=3.10 -y
conda activate mcp-agentic
pip install uv
uv pip install -r requirements.txt
