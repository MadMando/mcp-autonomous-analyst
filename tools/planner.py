# tools/planner.py

import os
import requests

import pandas as pd

from mcp.server.fastmcp import FastMCP
from tools.summarizer import summarize_outliers, describe_dataset
from tools.synthetic_data import generate_synthetic_data
from tools.outlier_detection import detect_outliers
from tools.plotter import plot_outliers
from tools.vector_store import log_to_chromadb


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:1b"


def call_ollama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "temperature": 0.3,
            "stream": False,
        }
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def agentic_decision_loop() -> str:
    # Step 1: Generate or load data
    data_path = "data/generated_data.csv"
    if not os.path.exists(data_path):
        df = generate_synthetic_data()
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    # Step 2: Outlier detection
    df = detect_outliers(df, features=["feature_1", "feature_2"])
    df.to_csv(data_path, index=False)

    # Step 3: Get summary and stats
    outlier_summary = summarize_outliers(df)
    stats_summary = describe_dataset(df)

    # Step 4: Create prompt to determine what to do next
    planner_prompt = (
        "You are an autonomous data analyst. Based on the outlier analysis below, "
        "decide on next actions.\n\n"
        f"--- Outlier Summary ---\n{outlier_summary}\n\n"
        f"--- Data Stats ---\n{stats_summary}\n\n"
        "Recommend the next steps such as re-analyzing with a different threshold, "
        "removing outliers, or logging results."
    )

    recommendation = call_ollama(planner_prompt)

    # Step 5: (Optionally) log results
    log_to_chromadb(df)

    return f"\n🔍 RECOMMENDATION:\n{recommendation}"


# MCP registration
mcp = FastMCP("AutonomousPlanner", stateless_http=True)


@mcp.tool(description="Run the full agentic loop and get next-step recommendation.")
def plan_and_recommend() -> str:
    return agentic_decision_loop()
