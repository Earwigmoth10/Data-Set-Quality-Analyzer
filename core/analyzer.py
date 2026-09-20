import checks

def _grade(score):
    for cutoff, g in [(90, "A"), (75, "B"), (60, "C"), (40, "D")]:
        if score >= cutoff:
            return g
    return "F"

def analyze(df, target=None):
    target = target or checks.guess_target(df)
    n_num = max(1, df.select_dtypes(include="number").shape[1])

    missing  = checks.check_missing(df)
    dups     = checks.check_duplicates(df)
    outliers = checks.check_outliers(df)
    zeros    = checks.check_suspicious_zeros(df)
    const    = checks.check_constant(df)
    corr     = checks.check_correlation(df)
    corr["pairs"] = [q for q in corr["pairs"] if target not in (q["col_a"], q["col_b"])]
    imb      = checks.check_imbalance(df, target)

    # the target's imbalance is reported separately, not as a "useless column"
    const["near_constant"].pop(target, None)

    # ---- penalties (each capped, total max = 100) ----
    mean_out = sum(c["percent"] for c in outliers["columns"].values()) / n_num
    sev = {"balanced": 0, "mild": 3, "moderate": 8, "severe": 15}
    penalties = {
        "missing":          min(25, missing["percent_of_cells"] * 5),
        "duplicates":       min(15, dups["percent"] * 0.75),
        "outliers":         min(15, mean_out * 3),
        "suspicious_zeros": min(10, sum(c["percent"] for c in zeros["columns"].values()) / 5),
        "imbalance":        sev.get(imb.get("severity"), 0),
        "useless_columns":  min(10, 5 * len(const["constant"])
                                   + 3 * len(const["near_constant"])
                                   + 2 * len(const["id_like"])),
        "correlation":      min(10, 2 * len(corr["pairs"])),
    }
    score = round(100 - sum(penalties.values()), 1)

    # ---- human-readable issues ("review", never "remove") ----
    issues = []
    if missing["total_missing"]:
        issues.append(f"Missing values in {missing['columns_affected']} column(s) "
                      f"({missing['percent_of_cells']}% of cells).")
    if dups["duplicate_rows"]:
        issues.append(f"{dups['duplicate_rows']} duplicate rows ({dups['percent']}%). "
                      "Please review: repeated rows can be legitimate.")
    if outliers["columns_affected"]:
        issues.append(f"Statistical outliers in {outliers['columns_affected']} column(s). "
                      "Please review before removing.")
    if zeros["columns_affected"]:
        issues.append("Suspicious zeros (possibly hidden missing values) in: "
                      + ", ".join(zeros["columns"]) + ".")
    if imb.get("severity") in ("moderate", "severe"):
        issues.append(f"Class imbalance in '{imb['target']}' is {imb['severity']} "
                      f"(ratio {imb['majority_minority_ratio']}:1).")
    if const["constant"] or const["near_constant"] or const["id_like"]:
        issues.append("Low-value columns: "
                      + ", ".join(const["constant"] + list(const["near_constant"]) + const["id_like"]) + ".")
    if corr["pairs"]:
        issues.append(f"{len(corr['pairs'])} highly correlated feature pair(s).")

    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "target": target,
        "score": score,
        "grade": _grade(score),
        "penalties": {k: round(v, 1) for k, v in penalties.items()},
        "issues": issues,
        "details": {"missing": missing, "duplicates": dups, "outliers": outliers,
                    "suspicious_zeros": zeros, "constant": const,
                    "correlation": corr, "imbalance": imb},
    }
