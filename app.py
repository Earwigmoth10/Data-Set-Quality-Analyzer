import os
import sys
import traceback

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(BASE_DIR, "core")
MODEL_PATH = os.path.join(BASE_DIR, "model", "quality_model.pkl")

# core/checks.py, cleaner.py, analyzer.py and features.py import each other with
# `import checks`, not `from . import checks` — so core/ has to sit directly on
# sys.path (not be imported as a package) for those imports to resolve.
sys.path.insert(0, CORE_DIR)

import analyzer  # noqa: E402
import features  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB upload cap

ALLOWED_EXTENSIONS = {".csv"}

# Load the trained model once at startup. If it's missing or fails to load,
# the app keeps working and just falls back to the rule-based score.
_model_bundle = None
_model_load_error = None
try:
    _model_bundle = joblib.load(MODEL_PATH)
except Exception as exc:
    _model_load_error = str(exc)
    print(f"[startup] Could not load {MODEL_PATH}: {exc}", file=sys.stderr)


def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


def predict_ml_score(df, target):
    """Predict a 0-100 quality score with the trained RandomForest."""
    if _model_bundle is None:
        raise RuntimeError(_model_load_error or "model not loaded")

    feats = features.meta_features(df, target)
    feature_names = _model_bundle["feature_names"]
    row = [[feats[name] for name in feature_names]]
    X = pd.DataFrame(row, columns=feature_names)

    pred = float(_model_bundle["model"].predict(X)[0])
    return max(0.0, min(100.0, pred))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file was sent."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file was selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Please upload a .csv file."}), 400

    try:
        df = pd.read_csv(file)
    except Exception:
        return jsonify({
            "error": "Couldn't read that as a CSV. Check the formatting and try again."
        }), 400

    if df.shape[0] == 0 or df.shape[1] == 0:
        return jsonify({"error": "That CSV appears to be empty."}), 400

    try:
        result = analyzer.analyze(df)
    except Exception:
        traceback.print_exc()
        return jsonify({
            "error": "Something went wrong while analyzing this file."
        }), 500

    # Prefer the trained model's score over the hand-tuned formula; fall back
    # to the rule-based score (and keep the app usable) if the model can't run.
    try:
        ml_score = predict_ml_score(df, result["target"])
        result["score"] = round(ml_score, 1)
        result["grade"] = analyzer._grade(ml_score)
        result["score_source"] = "model"
    except Exception:
        result["score_source"] = "rule_based"

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
