import os
import requests
import pandas as pd

from tools.summarizer import summarize_outliers, describe_dataset
from tools.synthetic_data import generate_synthetic_data
from tools.outlier_detection import detect_outliers
from tools.plotter import plot_outliers
from tools.vector_store import log_to_chromadb

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:1b"


def call_ollama(prompt: str) -> str:
    """Call the local LLM via Ollama with a structured prompt."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "temperature": 0.3,
            "stream": False,
        },
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def plan_and_recommend(input: str = "") -> str:
    """End-to-end agentic analysis with LLM recommendation."""
    data_path = "data/generated_data.csv"

    if not os.path.exists(data_path):
        df = generate_synthetic_data()
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    df = detect_outliers(df, features=["feature_1", "feature_2"])
    df.to_csv(data_path, index=False)

    outlier_summary = summarize_outliers(df)
    stats_summary = describe_dataset(df)

    planner_prompt = (
        "You are an autonomous data analyst. Based on the outlier analysis and data stats below, "
        "recommend the next course of action.\n\n"
        f"--- Outlier Summary ---\n{outlier_summary}\n\n"
        f"--- Data Stats ---\n{stats_summary}\n\n"
        "What should be done next? Options may include removing outliers, tuning the threshold, "
        "running another model, or logging findings."
    )

    recommendation = call_ollama(planner_prompt)
    log_to_chromadb(df)

    return (
        f"📊 Outlier Summary:\n{outlier_summary}\n\n"
        f"📈 Dataset Stats:\n{stats_summary}\n\n"
        f"🤖 LLM Recommendation:\n{recommendation}"
    )
