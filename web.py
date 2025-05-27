import os
import pandas as pd
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

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
                    <li><strong>generate_data:</strong> Synthetic data creation (No LLM)</li>
                    <li><strong>analyze_outliers:</strong> Mahalanobis anomaly detection (No LLM)</li>
                    <li><strong>plot_results:</strong> Inlier/outlier visualization (No LLM)</li>
                    <li><strong>summarize_results:</strong> Outlier analysis via llama3.2:1b</li>
                    <li><strong>summarize_data_stats:</strong> Stats to natural language via LLM</li>
                    <li><strong>log_results_to_vector_store:</strong> Archive results (ChromaDB)</li>
                    <li><strong>search_logs:</strong> Semantic log recall (ChromaDB + optional LLM)</li>
                    <li><strong>autonomous_plan:</strong> 🔁 Auto-planner that recommends next actions</li>
                </ul>
                <form action="/plan" method="get">
                    <button type="submit">🚀 Run Autonomous Planning</button>
                </form>
            </div>

            <div class="section">
                <h2>🧠 3. Agentic Behaviors</h2>
                <ul>
                    <li><strong>Autonomy:</strong> Agent determines what to do next based on data</li>
                    <li><strong>LLM Reasoning:</strong> Uses llama3.2:1b to interpret and decide</li>
                    <li><strong>Memory:</strong> Saves and retrieves insights via ChromaDB</li>
                    <li><strong>Tool Use:</strong> Uses MCP to invoke modular tools</li>
                </ul>
                <p><strong>LLM:</strong> llama3.2:1b via Ollama</p>
                <p><strong>Mode:</strong> Single-shot, temperature=0.1</p>
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
                <head><title>Analysis Results</title></head>
                <body>
                    <h1>🔍 Analysis Results</h1>
                    <div class="section"><h2>🖼️ Outlier Plot</h2><img src="/static/plot.png"></div>
                    <div class="section"><h2>📌 Summary (via llama3.2:1b)</h2><pre>{outlier_summary.content[0].text}</pre></div>
                    <div class="section"><h2>📈 Stats (via llama3.2:1b)</h2><pre>{stats_summary.content[0].text}</pre></div>
                </body>
            </html>
            """


@app.get("/plan", response_class=HTMLResponse)
async def run_autonomous_plan():
    try:
        async with streamablehttp_client("http://localhost:8000/mcp") as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("autonomous_plan")
                message = result.content[0].text if result.content else "No response from LLM."
    except Exception as e:
        message = f"Error running autonomous_plan: {str(e)}"

    return f"""
    <html>
        <head><title>Autonomous Plan Output</title></head>
        <body>
            <h1>🧠 Autonomous Pipeline Execution</h1>
            <p><strong>LLM Recommendation:</strong></p>
            <pre>{message}</pre>
            <p><a href="/">← Back to Dashboard</a></p>
        </body>
    </html>
    """
