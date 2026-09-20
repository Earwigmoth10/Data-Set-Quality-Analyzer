# Dataset Quality Analyzer

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Backend-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

Drop in a CSV, get back a letter grade. The app runs a set of data-quality
checks against an uploaded dataset — missing values, duplicates, outliers,
suspicious zeros, class imbalance, useless columns, correlated features —
and combines them into a single score, using a model trained to predict
how much a dataset's quality actually affects downstream model accuracy.
<img width="1312" height="1199" alt="ChatGPT Image Sep 20, 2026, 10_22_31 AM" src="https://github.com/user-attachments/assets/9c272a83-c3cc-4dde-8ae2-b119c3e6396b" />

## Screenshot

<!-- Add your screenshot to a `screenshots/` folder and update the path below -->
<img width="741" height="408" alt="qw" src="https://github.com/user-attachments/assets/6a51c7d6-4351-4812-947e-58defa7192a7" />
<img width="727" height="386" alt="qq" src="https://github.com/user-attachments/assets/52179f32-b465-4cfb-976e-de47a715095f" />
<img width="716" height="295" alt="qqq" src="https://github.com/user-attachments/assets/083326e9-47f3-454f-83aa-7a5bf1ca9400" />


## Overview

Feed the app a raw CSV and it answers three questions:

1. **How good is this dataset?** — a 0–100 score and an A–F letter grade.
2. **Where are the points being lost?** — a per-category penalty breakdown
   (missing values, duplicates, outliers, etc.).
3. **What should I look at first?** — a plain-language list of issues,
   phrased as things to review rather than instructions to blindly delete.

## Features

- Drag-and-drop CSV upload with client-side validation
- Rule-based quality checks: missing values (including hidden placeholders
  like `"?"`, `"NA"`, `"n/a"`), duplicate rows, statistical outliers (IQR
  method), suspicious zeros, constant/near-constant/ID-like columns,
  correlated feature pairs, and class imbalance
- A trained RandomForest model that predicts the overall quality score from
  those checks, cross-validated across nine real-world datasets
- Automatic fallback to the rule-based score if the trained model fails to
  load, so the app never breaks
- Clear error handling for empty files, non-CSV uploads, and malformed data

## How it works

```
CSV upload
    │
    ▼
core/checks.py        →  raw measurements (missing %, outlier %, ...)
    │
    ▼
core/analyzer.py      →  penalty breakdown + human-readable issues
    │
    ▼
core/features.py      →  measurements condensed into 11 model features
    │
    ▼
model/quality_model.pkl  →  predicted 0–100 score (overrides the rule-based one)
    │
    ▼
JSON response → rendered as the report in the browser
```

`core/cleaner.py` is included but not yet wired into the app — it can turn
the same diagnostics into an actual cleaned CSV (imputation, dropping
useless/correlated columns, capping outliers) and is a natural next
feature to expose through the UI.

## Tech stack

| Layer      | Technology                              |
|------------|------------------------------------------|
| Backend    | Flask (Python)                          |
| Model      | scikit-learn `RandomForestRegressor`     |
| Data       | pandas, numpy                           |
| Frontend   | Vanilla HTML / CSS / JavaScript          |
| Testing    | pytest                                  |

## Project structure

```
dataset-quality-app/
├── app.py                   # Flask entrypoint — serves the UI and /analyze
├── requirements.txt
├── model/
│   └── quality_model.pkl    # trained RandomForest + metadata bundle
├── core/
│   ├── checks.py            # individual quality checks
│   ├── cleaner.py           # turns checks into an actual cleaned dataframe
│   ├── analyzer.py          # combines checks into a score + issue list
│   └── features.py          # condenses checks into model input features
├── training/
│   ├── generate_data.py     # builds the training set via corruption recipes
│   └── train_model.py       # trains and validates quality_model.pkl
├── static/
│   ├── css/style.css
│   └── js/main.js
├── templates/
│   └── index.html
├── tests/
│   └── test_project.py
└── uploads/                  # scratch space for uploaded files (gitignored)
```

## Getting started

**Requirements:** Python 3.10+

```bash
git clone <your-repo-url>
cd dataset-quality-app
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser and drop in a CSV.

## Usage

1. Drag a `.csv` file onto the dropzone, or click to browse.
2. The app reads the file, runs the checks, and returns a report:
   - **Grade circle** — A–F, colored by score band (green/amber/red)
   - **Score** — 0–100, with a note on whether it came from the trained
     model or the rule-based fallback
   - **Stats** — row count, column count, and the detected target column
   - **Penalty breakdown** — how many points were lost in each category
   - **Notes in the margin** — the specific issues found, in plain language
3. Click **Analyze another file** to reset and try a different dataset.

### Grade scale

| Grade | Score range |
|-------|-------------|
| A     | 90–100      |
| B     | 75–89.9     |
| C     | 60–74.9     |
| D     | 40–59.9     |
| F     | below 40    |

## API reference

### `POST /analyze`

Multipart form upload, field name `file`, containing a single `.csv`.

**Success — `200 OK`**
```json
{
  "shape": { "rows": 308, "columns": 9 },
  "target": "churned",
  "score": 91.2,
  "grade": "A",
  "score_source": "model",
  "penalties": {
    "missing": 4.3,
    "duplicates": 0.4,
    "outliers": 0.8,
    "suspicious_zeros": 2.0,
    "imbalance": 8,
    "useless_columns": 5,
    "correlation": 2
  },
  "issues": [
    "Missing values in 1 column(s) (0.87% of cells).",
    "Class imbalance in 'churned' is moderate (ratio 5.09:1)."
  ],
  "details": { "...": "full check output, per category" }
}
```

**Error — `400` / `500`**
```json
{ "error": "Please upload a .csv file." }
```

## The quality model

`model/quality_model.pkl` is a bundle (loaded with `joblib`) containing:

| Key            | Contents                                                    |
|----------------|--------------------------------------------------------------|
| `model`        | trained `RandomForestRegressor`                              |
| `feature_names`| the 11 input features, in the order the model expects        |
| `band_edges`   | `{"good": 95, "fair": 85}` — internal Good/Fair/Poor cutoffs  |
| `trained_on`   | dataset names used in training/validation                    |
| `cv_metrics`   | leave-one-dataset-out validation results                     |

Validation results (leave-one-dataset-out, across 9 datasets):

| Metric                | Model  | Baseline |
|------------------------|--------|----------|
| MAE                    | 3.18   | 5.45     |
| R²                     | 0.643  | -0.023   |
| Spearman correlation   | 0.389  | —        |
| Band accuracy (Good/Fair/Poor) | 0.822 | 0.758 |

The Spearman correlation is modest, meaning the model ranks dataset
quality only loosely — treat the score as a useful signal, not a precise
metric.

### Retraining

```bash
cd training
python generate_data.py   # rebuilds training_data.csv from datasets/
python train_model.py     # retrains and re-saves quality_model.pkl
```

## Testing

```bash
pip install pytest
pytest tests -q
```

## Datasets for testing

A handful of small CSVs you can pull straight from GitHub to try the app
against — each exercises a different kind of quality issue.

| Dataset  | Good for testing                | Link |
|----------|----------------------------------|------|
| Titanic  | Missing values, mixed types      | https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv |
| Penguins | A few missing values             | https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv |
| Iris     | Clean — a "good quality" baseline| https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv |
| Diamonds | Large, has outliers               | https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv |
| Tips     | Small, mixed categorical/numeric | https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv |

For bigger, messier, more realistic datasets:

| Dataset               | Why it's useful                    | Link |
|------------------------|-------------------------------------|------|
| Adult Income           | Missing values, class imbalance     | https://archive.ics.uci.edu/dataset/2/adult |
| Wine Quality           | Correlated features, outliers       | https://archive.ics.uci.edu/dataset/186/wine+quality |
| Heart Disease          | Missing values, medical data        | https://archive.ics.uci.edu/dataset/45/heart+disease |
| Pima Indians Diabetes  | Hidden missing values (stored as 0) | https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database |
| Credit Card Fraud      | Extreme class imbalance             | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| Telco Customer Churn   | Duplicates and messy types          | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |

## Known limitations

- **Target detection** only looks at the last column of the CSV; if the
  label column isn't last, it won't be picked up automatically and class
  imbalance won't be scored.
- **Large files** are read and analyzed synchronously — there's no
  progress indicator or size limit beyond Flask's 20 MB upload cap.
- **`core/cleaner.py`** is not yet exposed through the UI or API.

## Roadmap

| Phase | What it involves | Output | Status |
|-------|-------------------|--------|--------|
| 1. Datasets | Collect public CSVs (UCI, Kaggle: Titanic, Adult, Diabetes, Wine, etc.) | `data/` folder | ✅ Done |
| 2. Core checks | Each quality check as its own function | `checks.py` | ✅ Done |
| 3. Cleaning | Impute, drop duplicates, cap outliers, drop constant/correlated columns | `cleaner.py` | ✅ Done (not yet wired into the app) |
| 4. ML layer | Synthetic corruption → meta-features → trained quality-score model | `quality_model.pkl` | ✅ Done |
| 5. Validation | Leave-one-dataset-out CV, MAE/R²/Spearman, band accuracy | Metrics in `cv_metrics` | ✅ Done |
| 6. Diagrams | Feature importance / residual plots from training | PNGs in `figures/` | ✅ Done (basic set — heatmaps/boxplots not yet added) |
| 7. Flask + frontend | Upload page → analyze → show report | Working web app | ✅ Done |
| 8. Deployment | Render / Railway / Hugging Face Spaces | Live link | ⬜ Not started |

Near-term additions on top of Phase 8:
- [ ] "Download cleaned CSV" using `core/cleaner.py`
- [ ] Let the user manually select the target column
- [ ] Missing-value heatmap, correlation heatmap, class-distribution chart in the report
- [ ] Progress indicator for large file uploads
- [ ] Isolation Forest as a second anomaly-detection model alongside the quality-score regressor

## License

Add your license of choice here (MIT, Apache 2.0, etc.).

## Author

**Laiba Aamir**
