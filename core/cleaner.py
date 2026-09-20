import numpy as np
import pandas as pd
import checks
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def _placeholders_to_nan(s):
    txt = s.astype(str).str.strip().str.lower()
    return s.where(~txt.isin(checks.PLACEHOLDERS) & s.notna())


def _drop_correlated(df, target, threshold):
    """Greedy: drop the weaker column of the most correlated pair, repeat."""
    dropped = []
    target_numeric = target in df.columns and pd.api.types.is_numeric_dtype(df[target])
    while True:
        num = df.select_dtypes(include="number").drop(columns=[target], errors="ignore")
        if num.shape[1] < 2:
            break
        corr = num.corr().abs().fillna(0)
        corr = corr.where(~np.eye(len(corr), dtype=bool), 0)
        a, b = corr.stack().idxmax()
        if corr.loc[a, b] < threshold:
            break
        if target_numeric:
            # keep the column more related to the target
            drop = b if abs(df[a].corr(df[target])) >= abs(df[b].corr(df[target])) else a
        else:
            # keep the less redundant column
            drop = a if corr[a].mean() >= corr[b].mean() else b
        dropped.append((drop, a if drop == b else b, round(float(corr.loc[a, b]), 3)))
        df = df.drop(columns=[drop])
    return df, dropped


def _knn_impute(df, cols, k=5):
    """Scale, impute from the k most similar rows, then unscale."""
    scaler = StandardScaler()
    X = KNNImputer(n_neighbors=k).fit_transform(scaler.fit_transform(df[cols]))
    out = df.copy()
    out[cols] = scaler.inverse_transform(X)
    return out


def clean(df, target=None, zero_cols=(), drop_duplicates=False,
          cap_outliers=False, corr_threshold=0.9, max_missing_pct=60,
          imputer="median", warn_missing_pct=30, impute=True):
    df = df.copy()
    log = []

    # 1. hidden missing values and numbers stored as text
    for col in df.select_dtypes(exclude="number").columns:
        if pd.api.types.is_bool_dtype(df[col]):
            continue
        s = _placeholders_to_nan(df[col])
        hidden = int(s.isna().sum() - df[col].isna().sum())
        num = pd.to_numeric(s, errors="coerce")
        if s.notna().sum() and num.notna().sum() / s.notna().sum() >= 0.95:
            df[col] = num
            log.append(f"'{col}': converted text to numeric ({hidden} hidden missing -> NaN)")
        elif hidden:
            df[col] = s
            log.append(f"'{col}': {hidden} placeholder values -> NaN")

    # 2. suspicious zeros, only for columns the user confirmed
    for col in zero_cols:
        n = int((df[col] == 0).sum())
        df[col] = df[col].replace(0, np.nan)
        log.append(f"'{col}': {n} zeros treated as missing")

    # 3. duplicates (opt-in)
    if drop_duplicates:
        n = int(df.duplicated().sum())
        df = df.drop_duplicates()
        log.append(f"Dropped {n} duplicate rows")

    # 4. useless columns (never the target)
    const = checks.check_constant(df)
    useless = (set(const["constant"]) | set(const["near_constant"]) | set(const["id_like"])) - {target}
    if useless:
        df = df.drop(columns=sorted(useless))
        log.append(f"Dropped useless columns: {sorted(useless)}")

    # 5. columns that are mostly empty
    pct = df.isna().mean() * 100
    too_empty = [c for c in df.columns if pct[c] > max_missing_pct and c != target]
    if too_empty:
        df = df.drop(columns=too_empty)
        log.append(f"Dropped columns with >{max_missing_pct}% missing: {too_empty}")

    # 6. impute (rows with a missing target are dropped, never imputed)
    if target in df.columns and df[target].isna().any():
        n = int(df[target].isna().sum())
        df = df.dropna(subset=[target])
        log.append(f"Dropped {n} rows with missing target")
    knn_used = False
    for col in df.columns:
        if col == target or not impute or not df[col].isna().any():
            continue
        n = int(df[col].isna().sum())
        share = 100 * n / len(df)
        if share > warn_missing_pct:
            log.append(f"WARNING '{col}': {share:.1f}% missing, imputed values are unreliable")
        if pd.api.types.is_numeric_dtype(df[col]):
            if imputer == "knn":
                knn_used = True
                log.append(f"'{col}': {n} missing -> KNN estimate")
            else:
                df[col] = df[col].fillna(df[col].median())
                log.append(f"'{col}': {n} missing -> median")
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
            log.append(f"'{col}': {n} missing -> most frequent value")
    if knn_used:
        num_cols = [c for c in df.select_dtypes(include="number").columns if c != target]
        df = _knn_impute(df, num_cols)

    # 7. cap outliers (opt-in)
    if cap_outliers:
        for col in df.select_dtypes(include="number").columns:
            if col == target or df[col].nunique() <= 2:
                continue
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n = int(((df[col] < lo) | (df[col] > hi)).sum())
            if n:
                df[col] = df[col].clip(lo, hi)
                log.append(f"'{col}': {n} outliers capped to [{lo:.2f}, {hi:.2f}]")

    # 8. correlated features
    df, dropped = _drop_correlated(df, target, corr_threshold)
    for drop, keep, c in dropped:
        log.append(f"Dropped '{drop}' (correlation {c} with '{keep}')")

    return df, log
