# tools/plotter.py

import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_outliers(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: str = "static/plot.png"
) -> str:
    plt.figure(figsize=(8, 6))

    # Plot inliers and outliers
    inliers = df[df["outlier"] == "inlier"]
    outliers = df[df["outlier"] == "outlier"]

    plt.scatter(inliers[x_col], inliers[y_col], c="blue", label="Inlier", alpha=0.6)
    plt.scatter(outliers[x_col], outliers[y_col], c="red", label="Outlier", alpha=0.8)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("Outlier Detection")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    return output_path
