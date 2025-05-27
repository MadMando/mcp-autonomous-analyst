# web.py (MCP client with tool descriptions and agentic explanations)

import os

import pandas as pd
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


# FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

DATA_PATH = "data/generated_data.csv"
PLOT_PATH = "static/plot.png"


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return """
    <html>
        <head>
            <title>Autonomous Analyst</title>
            <style>
                body { font-family: Arial; margin: 2em; }
                img { max-width: 600px; border: 1px solid #ccc; }
                .section { margin-bottom: 2em; }
                h2 { border-bottom: 2px solid #eee; padding-bottom: 0.2em; }
                pre { background: #f9f9f9; padding: 1em; border: 1px solid #ddd; }
                form { margin-bottom: 2em; }
                input, select { margin-right: 1em; }
            </style>
        </head>
        <body>
            <h1>📊 Autonomous Analyst Dashboard</h1>

            <div class="section">
                <h2>📁 1. Data Selection</h2>
                <form action="/analyze" method="post">
                    <button name="action" value="generate">Generate Synthetic Data</button>
                </form>
                <form action="/analyze" method="post" enctype="multipart/form-data">
                    <input type="file" name="file">
                    <button type="submit" name="action" value="upload">Upload Your Dataset</button>
                </form>
            </div>

            <div class="section">
                <h2>⚙️ 2. Toolchain (MCP Tools)</h2>
                <ul>
                    <li><strong>generate_data:</strong> Synthesizes realistic datasets using normal and categorical sampling. <em>(No LLM)</em></li>
                    <li><strong>analyze_outliers:</strong> Applies Mahalanobis distance for anomaly detection. <em>(No LLM)</em></li>
                    <li><strong>plot_results:</strong> Generates a Matplotlib-based scatter plot of inliers vs. outliers. <em>(No LLM)</em></li>
                    <li><strong>summarize_results:</strong> Uses <code>llama3.2:1b</code> via Ollama to explain outliers and suggest next steps.</li>
                    <li><strong>summarize_data_stats:</strong> Uses <code>llama3.2:1b</code> to interpret statistical patterns in natural language.</li>
                    <li><strong>log_results_to_vector_store:</strong> Persists summaries to ChromaDB for future reference. <em>(No LLM)</em></li>
                    <li><strong>search_logs:</strong> Retrieves relevant sessions using semantic similarity. <em>(Uses ChromaDB, optional LLM)</em></li>
                </ul>
            </div>

            <div class="section">
                <h2>🧠 3. Agentic Behaviors</h2>
                <ul>
                    <li><strong>Autonomy:</strong> Chooses workflows based on data source (synthetic or uploaded).</li>
                    <li><strong>Reasoning:</strong> Summarizes complex insights using LLMs for interpretability.</li>
                    <li><strong>Memory:</strong> Logs and retrieves analysis sessions using a vector database.</li>
                    <li><strong>Tool Use:</strong> All processing is driven by callable tools exposed via the MCP protocol.</li>
                </ul>
                <p><strong>LLM Used:</strong> llama3.2:1b via local Ollama instance</p>
                <p><strong>Inference Mode:</strong> Single-shot, deterministic (temperature = 0.1)</p>
            </div>
        </body>
    </html>
    """


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(action: str = Form(...), file: UploadFile = None):
    os.makedirs("data", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    async with streamablehttp_client("http://localhost:8000/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            if action == "generate":
                await session.call_tool("generate_data")
            elif action == "upload" and file:
                df = pd.read_csv(file.file)
                df.to_csv(DATA_PATH, index=False)
            else:
                return HTMLResponse("<h3>Error: No data provided.</h3>", status_code=400)

            await session.call_tool("analyze_outliers", {"x_col": "feature_1", "y_col": "feature_2"})
            await session.call_tool("plot_results", {"x_col": "feature_1", "y_col": "feature_2"})
            outlier_summary = await session.call_tool("summarize_results")
            stats_summary = await session.call_tool("summarize_data_stats")

            return f"""
            <html>
                <head>
                    <title>Autonomous Analyst Results</title>
                    <style>
                        body {{ font-family: Arial; margin: 2em; }}
                        img {{ max-width: 600px; border: 1px solid #ccc; }}
                        .section {{ margin-bottom: 2em; }}
                        h2 {{ border-bottom: 2px solid #eee; padding-bottom: 0.2em; }}
                        pre {{ background: #f9f9f9; padding: 1em; border: 1px solid #ddd; }}
                    </style>
                </head>
                <body>
                    <h1>🔍 Analysis Results</h1>

                    <div class="section">
                        <h2>🖼️ Outlier Visualization</h2>
                        <img src="/static/plot.png" alt="Outlier Plot">
                    </div>

                    <div class="section">
                        <h2>📌 Outlier Summary (via llama3.2:1b)</h2>
                        <pre>{outlier_summary.content[0].text}</pre>
                    </div>

                    <div class="section">
                        <h2>📈 Dataset Description (via llama3.2:1b)</h2>
                        <pre>{stats_summary.content[0].text}</pre>
                    </div>
                </body>
            </html>
            """
