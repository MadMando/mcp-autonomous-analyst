# tools/synthetic_data.py

import numpy as np
import pandas as pd


def generate_synthetic_data(rows: int = 3000) -> pd.DataFrame:
    np.random.seed()
    df = pd.DataFrame({
        "feature_1": np.random.normal(50, 10, rows),
        "feature_2": np.random.normal(30, 5, rows),
        "category": np.random.choice(["A", "B", "C"], rows)
    })

    outliers = pd.DataFrame({
        "feature_1": np.random.normal(150, 5, 20),
        "feature_2": np.random.normal(90, 5, 20),
        "category": ["Outlier"] * 20
    })

    return pd.concat([df, outliers], ignore_index=True)
