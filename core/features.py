import numpy as np
import pandas as pd
import checks

FEATURE_NAMES = [
    "n_rows", "n_cols", "missing_pct", "max_col_missing_pct", "duplicate_pct",
    "outlier_mean_pct", "outlier_cols_frac", "zero_mean_pct", "constant_frac",
    "id_like_frac", "corr_pairs_per_col", "imbalance_log_ratio", "minority_pct",
]


def meta_features(df, target):
    """Summarise the six quality checks as one numeric row (input of the ML model)."""
    n_rows, n_cols = df.shape
    n_num = max(1, df.select_dtypes(include="number").shape[1])
    f = {"n_rows": n_rows, "n_cols": n_cols}

    m = checks.check_missing(df)
    f["missing_pct"] = m["percent_of_cells"]
    f["max_col_missing_pct"] = max((c["percent"] for c in m["columns"].values()), default=0.0)
    f["duplicate_pct"] = checks.check_duplicates(df)["percent"]

    o = checks.check_outliers(df)
    f["outlier_mean_pct"] = sum(c["percent"] for c in o["columns"].values()) / n_num
    f["outlier_cols_frac"] = o["columns_affected"] / n_num

    z = checks.check_suspicious_zeros(df)
    f["zero_mean_pct"] = sum(c["percent"] for c in z["columns"].values()) / n_num

    c = checks.check_constant(df)
    c["near_constant"].pop(target, None)
    f["constant_frac"] = (len(c["constant"]) + len(c["near_constant"])) / n_cols
    f["id_like_frac"] = len(c["id_like"]) / n_cols

    pairs = [q for q in checks.check_correlation(df)["pairs"]
             if target not in (q["col_a"], q["col_b"])]
    f["corr_pairs_per_col"] = len(pairs) / n_cols

    imb = checks.check_imbalance(df, target)
    f["imbalance_log_ratio"] = float(np.log(imb.get("majority_minority_ratio", 1.0)))
    f["minority_pct"] = imb.get("minority_percent", 50.0)

    return {k: float(f[k]) for k in FEATURE_NAMES}
