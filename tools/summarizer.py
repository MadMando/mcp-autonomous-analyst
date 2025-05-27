import requests
import pandas as pd


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:1b"


def call_ollama(prompt: str) -> str:
    """Send a prompt to the local Ollama server."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "temperature": 0.1,
            "stream": False,
        },
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def summarize_outliers(df: pd.DataFrame) -> str:
    """Generate a technical summary and recommendation for detected outliers."""
    if "outlier" not in df.columns:
        return "Outlier analysis not found. Please run analyze_outliers first."

    inliers = int((df["outlier"] == "inlier").sum())
    outliers = int((df["outlier"] == "outlier").sum())

    prompt = (
        "You are a senior data analyst. The dataset was analyzed using "
        "Isolation Forest.\n\n"
        "Summary statistics:\n"
        f"- Inliers detected: {inliers}\n"
        f"- Outliers detected: {outliers}\n\n"
        "Provide a technical interpretation of what this implies about the "
        "dataset structure, potential data quality issues, or anomalies.\n\n"
        "Then recommend a course of action regarding these outliers. Should they "
        "be removed, investigated further, or used for model tuning? Justify your "
        "recommendation."
    )
    return call_ollama(prompt)


def describe_dataset(df: pd.DataFrame) -> str:
    """Generate a technical summary of numeric and categorical distributions."""
    numeric_desc = (
        df.select_dtypes(include="number")
        .describe()
        .round(2)
        .to_string()
    )

    categorical_cols = df.select_dtypes(include="object").columns
    category_summary = (
        "\n".join(
            f"{col}: {dict(df[col].value_counts().head(5))}"
            for col in categorical_cols
        )
        if not categorical_cols.empty
        else "No categorical columns."
    )

    prompt = (
        "You are a statistical analyst reviewing a structured dataset.\n\n"
        f"Numerical Feature Summary:\n{numeric_desc}\n\n"
        f"Categorical Feature Summary:\n{category_summary}\n\n"
        "Please identify key patterns, distributions, and any potential data "
        "skew or imbalance. Summarize these findings in a technical but concise "
        "report suitable for inclusion in a data quality assessment."
    )
    return call_ollama(prompt)
