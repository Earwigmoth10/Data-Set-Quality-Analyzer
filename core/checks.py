import pandas as pd

PLACEHOLDERS = {"", "?", "na", "n/a", "nan", "null", "none", "-"}

def check_missing(df):
    n_rows = len(df)
    columns = {}

    for col in df.columns:
        real = int(df[col].isnull().sum())
        hidden = 0
        if not pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].dropna().astype(str).str.strip().str.lower()
            hidden = int(s.isin(PLACEHOLDERS).sum())

        if real + hidden > 0:
            columns[col] = {
                "missing": real,
                "hidden_missing": hidden,
                "percent": round(100 * (real + hidden) / n_rows, 2),
            }

    total = sum(c["missing"] + c["hidden_missing"] for c in columns.values())
    return {
        "total_missing": total,
        "percent_of_cells": round(100 * total / df.size, 2),
        "columns_affected": len(columns),
        "columns": columns,
    }


def check_duplicates(df):
    mask = df.duplicated(keep="first")
    n = int(mask.sum())
    return {
        "duplicate_rows": n,
        "percent": round(100 * n / len(df), 2),
        "example_indices": df.index[mask][:5].tolist(),
    }


def check_outliers(df, factor=1.5):
    """IQR method: flag values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    columns = {}
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        if s.nunique() <= 2:          # skip binary columns
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:                  # no spread, IQR can't work
            continue
        low, high = q1 - factor * iqr, q3 + factor * iqr
        n = int(((s < low) | (s > high)).sum())
        if n > 0:
            columns[col] = {
                "outliers": n,
                "percent": round(100 * n / len(s), 2),
                "lower_bound": round(float(low), 3),
                "upper_bound": round(float(high), 3),
            }
    return {"columns_affected": len(columns), "columns": columns}


def check_suspicious_zeros(df):
    """Zeros in continuous columns that may really mean 'missing'."""
    columns = {}
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        if s.nunique() <= 20:         # skip counts/categories (e.g. Pregnancies)
            continue
        zeros = int((s == 0).sum())
        pct = 100 * zeros / len(s)
        nonzero = s[s != 0]
        if zeros > 0 and pct < 60 and nonzero.min() > 0:
            columns[col] = {"zeros": zeros, "percent": round(pct, 2)}
    return {"columns_affected": len(columns), "columns": columns}


def check_constant(df, near_threshold=0.99):
    """Constant, near-constant, and ID-like columns (useless for modelling)."""
    constant, near_constant, id_like = [], {}, []
    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0 or s.nunique() <= 1:
            constant.append(col)
            continue
        top_share = s.value_counts(normalize=True).iloc[0]
        if top_share >= near_threshold:
            near_constant[col] = round(100 * float(top_share), 2)
        # ID-like: every value unique in a text or integer column
        is_text = not pd.api.types.is_numeric_dtype(s)
        is_int = pd.api.types.is_integer_dtype(s)
        if s.nunique() == len(s) and (is_text or is_int) and len(s) > 20:
            id_like.append(col)
    return {"constant": constant, "near_constant": near_constant, "id_like": id_like}


def check_correlation(df, threshold=0.9):
    """Pairs of numeric columns with |correlation| >= threshold."""
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return {"pairs": []}
    corr = num.corr().abs()
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if pd.notna(v) and v >= threshold:
                pairs.append({"col_a": cols[i], "col_b": cols[j],
                              "correlation": round(float(v), 3)})
    pairs.sort(key=lambda p: -p["correlation"])
    return {"pairs": pairs}


def guess_target(df):
    """Guess the target column: last column if it looks like a class label."""
    last = df.columns[-1]
    if df[last].nunique() <= 10:
        return last
    return None


def check_imbalance(df, target=None):
    target = target or guess_target(df)
    if target is None or target not in df.columns:
        return {"target": None, "note": "no class column detected"}
    counts = df[target].value_counts()
    ratio = float(counts.iloc[0] / counts.iloc[-1])
    if ratio < 1.5:
        severity = "balanced"
    elif ratio < 3:
        severity = "mild"
    elif ratio < 10:
        severity = "moderate"
    else:
        severity = "severe"
    return {
        "target": target,
        "classes": {str(k): int(v) for k, v in counts.items()},
        "majority_minority_ratio": round(ratio, 2),
        "minority_percent": round(100 * float(counts.iloc[-1] / counts.sum()), 2),
        "severity": severity,
    }
