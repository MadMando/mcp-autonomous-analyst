# server.py

import logging

import pandas as pd

from mcp.server.fastmcp import FastMCP
from tools.synthetic_data import generate_synthetic_data
from tools.outlier_detection import detect_outliers
from tools.plotter import plot_outliers
from tools.summarizer import summarize_outliers, describe_dataset
from tools.vector_store import log_to_chromadb, query_recent_sessions
from tools.planner import plan_and_recommend


# Enable logging
logging.basicConfig(level=logging.INFO)
print("Loaded server.py")

mcp = FastMCP("AutonomousAnalyst", stateless_http=True)


@mcp.tool(description="Search past session logs for a topic.")
def search_logs(query: str = "outliers") -> str:
    return "\n\n".join(query_recent_sessions(query))


@mcp.tool(description="Generate synthetic data and save to disk.")
def generate_data() -> str:
    df = generate_synthetic_data()
    df.to_csv("data/generated_data.csv", index=False)
    return f"{len(df)} rows of synthetic data generated."


@mcp.tool(description="Detect outliers using Isolation Forest.")
def analyze_outliers(x_col: str = "feature_1", y_col: str = "feature_2") -> str:
    df = pd.read_csv("data/generated_data.csv")
    df = detect_outliers(df, [x_col, y_col])
    df.to_csv("data/generated_data.csv", index=False)
    return f"Outlier analysis complete. Columns: {x_col}, {y_col}."


@mcp.tool(description="Plot inliers and outliers and save plot.")
def plot_results(x_col: str = "feature_1", y_col: str = "feature_2") -> str:
    df = pd.read_csv("data/generated_data.csv")
    path = plot_outliers(df, x_col, y_col)
    return f"Plot saved to {path}"


@mcp.tool(description="Summarize outlier results using LLM.")
def summarize_results() -> str:
    df = pd.read_csv("data/generated_data.csv")
    return summarize_outliers(df)


@mcp.tool(description="Summarize descriptive stats using LLM.")
def summarize_data_stats() -> str:
    df = pd.read_csv("data/generated_data.csv")
    return describe_dataset(df)


@mcp.tool(description="Store summary to vector store.")
def log_results_to_vector_store() -> str:
    df = pd.read_csv("data/generated_data.csv")
    return log_to_chromadb(df)


@mcp.tool(description="Agentic recommendation based on full dataset analysis.")
def autonomous_plan() -> str:
    df = pd.read_csv("data/generated_data.csv")
    summary1 = summarize_outliers(df)
    summary2 = describe_dataset(df)
    recommendation = plan_and_recommend(df)
    log_to_chromadb(df)  # persist results
    return (
        f"📊 Outlier Summary:\n{summary1}\n\n"
        f"📈 Dataset Overview:\n{summary2}\n\n"
        f"🤖 Recommendation:\n{recommendation}"
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8001)
