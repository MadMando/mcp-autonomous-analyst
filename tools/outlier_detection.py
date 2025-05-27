# tools/outlier_detection.py

import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis


def detect_outliers(
    df: pd.DataFrame,
    features: list[str],
    threshold: float = 4.0
) -> pd.DataFrame:
    """
    Detect outliers in a DataFrame using Mahalanobis distance.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        features (list[str]): List of feature columns to consider.
        threshold (float): Mahalanobis distance threshold for flagging outliers.

    Returns:
        pd.DataFrame: The original DataFrame with
                      'mahalanobis_distance' and 'outlier' columns added.
    """
    data = df[features].copy()

    # Compute covariance matrix and its inverse
    cov_matrix = np.cov(data, rowvar=False)
    inv_cov_matrix = np.linalg.pinv(cov_matrix)
    mean_vector = data.mean().values

    # Calculate Mahalanobis distance for each row
    df["mahalanobis_distance"] = data.apply(
        lambda row: mahalanobis(row, mean_vector, inv_cov_matrix),
        axis=1
    )

    # Classify as outlier or inlier
    df["outlier"] = df["mahalanobis_distance"].apply(
        lambda dist: "outlier" if dist > threshold else "inlier"
    )

    return df
