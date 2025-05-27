# tools/vector_store.py

import os
import uuid

import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


CHROMA_DIR = "chroma_db"

# Use the new PersistentClient
client = chromadb.PersistentClient(path=CHROMA_DIR)

# Optional: create embedding function (if not using default server-side)
embedding_function = DefaultEmbeddingFunction()

# Initialize or get collection
collection = client.get_or_create_collection(
    name="analysis_logs",
    embedding_function=embedding_function
)


def log_to_chromadb(df: pd.DataFrame) -> str:
    if "outlier" not in df.columns:
        return "Data must be analyzed before logging. Run analyze_outliers first."

    summary = {
        "inliers": int((df["outlier"] == "inlier").sum()),
        "outliers": int((df["outlier"] == "outlier").sum()),
    }
    description = df.describe().to_string()

    doc = (
        f"Session Summary:\n"
        f"Inliers: {summary['inliers']}, Outliers: {summary['outliers']}\n\n"
        f"Stats:\n{description}"
    )

    uid = str(uuid.uuid4())
    collection.add(
        documents=[doc],
        metadatas=[{"source": "generated_data.csv"}],
        ids=[uid]
    )

    return f"Session logged in ChromaDB with ID: {uid}"


def query_recent_sessions(query: str, n_results: int = 3) -> list[str]:
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0] if results["documents"] else ["No relevant sessions found."]
