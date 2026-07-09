#!/usr/bin/env python
"""Run structured ML baselines for post-ICI AKI prediction."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


CHARLSON_COLS = [
    "HIV",
    "AIDS",
    "Cerebrovascular_Disease",
    "Congestive_Heart_Failure",
    "Myocardial_Infarction",
    "Peripheral_Vascular_Disease",
    "Chronic_Pulmonary_Disease",
    "Dementia",
    "Liver_Disease_Mild",
    "Liver_Disease_Moderate_Severe",
    "Malignancy",
    "Metastatic_Solid_Tumor",
    "Peptic_Ulcer_Disease",
    "Renal_Disease_Mild_Moderate",
    "Renal_Disease_Severe",
    "Rheumatic_Disease",
    "Hemiplegia_Paraplegia",
    "Diabetes_with_Chronic_Complications",
    "Diabetes_without_Chronic_Complications",
]

OUTCOMES = [
    ("aki_any", 180),
    ("aki_3mo", 90),
    ("aki_6mo", 180),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AKI structured ML baselines.")
    parser.add_argument("--denominator-csv", required=True)
    parser.add_argument("--baseline-features-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-type", default="auto", choices=["auto", "xgboost", "hist_gradient_boosting"])
    parser.add_argument("--class-weight", default="none", choices=["none", "balanced"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--include-regimen", action="store_true")
    parser.add_argument(
        "--min-any-control-followup-days",
        type=int,
        default=180,
        help="Minimum follow-up for no-AKI controls in the any-AKI endpoint.",
    )
    return parser.parse_args()


def as_bool_series(frame, evidence_col: str = "aki_evidence"):
    evidence = frame[evidence_col].fillna("none").astype(str).str.strip().str.lower()
    return evidence.ne("none") & evidence.ne("") & evidence.ne("nan")


def load_analysis_frame(denominator_csv: str | Path, baseline_features_csv: str | Path):
    import pandas as pd

    denom = pd.read_csv(denominator_csv)
    baseline = pd.read_csv(baseline_features_csv)
    label_cols = ["person_id", "aki_evidence", "days_to_aki", "followup_days", "ici_regimen"]
    labels = denom[[col for col in label_cols if col in denom.columns]].copy()
    frame = baseline.merge(labels, on="person_id", how="inner", validate="one_to_one")

    frame["days_to_aki"] = pd.to_numeric(frame.get("days_to_aki"), errors="coerce")
    frame["followup_days"] = pd.to_numeric(frame.get("followup_days"), errors="coerce")
    frame["aki_any"] = as_bool_series(frame)
    frame["aki_3mo"] = frame["aki_any"] & frame["days_to_aki"].le(90)
    frame["aki_6mo"] = frame["aki_any"] & frame["days_to_aki"].le(180)
    return frame


def feature_columns(frame, include_regimen: bool) -> tuple[list[str], list[str]]:
    numeric = ["age_at_index", "charlson_comorbidity_count"]
    numeric += [col for col in CHARLSON_COLS if col in frame.columns]
    numeric = [col for col in numeric if col in frame.columns]

    categorical = [col for col in ["gender", "race", "ethnicity"] if col in frame.columns]
    if include_regimen and "ici_regimen" in frame.columns:
        categorical.append("ici_regimen")
    return numeric, categorical


def choose_model(model_type: str, seed: int, y_train) -> tuple[str, Any]:
    if model_type in {"auto", "xgboost"}:
        try:
            from xgboost import XGBClassifier

            return (
                "xgboost",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_jobs=4,
                    random_state=seed,
                ),
            )
        except Exception:
            if model_type == "xgboost":
                raise

    from sklearn.ensemble import HistGradientBoostingClassifier

    return (
        "hist_gradient_boosting",
        HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.03,
            l2_regularization=0.01,
            random_state=seed,
        ),
    )


def make_pipeline(numeric_cols: list[str], categorical_cols: list[str], model):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", encoder),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def compute_metrics(labels, probs, threshold: float = 0.5) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    preds = probs >= threshold
    labels_bool = labels.astype(bool)
    tp = int((labels_bool & preds).sum())
    tn = int(((~labels_bool) & (~preds)).sum())
    fp = int(((~labels_bool) & preds).sum())
    fn = int((labels_bool & (~preds)).sum())
    pos_probs = probs[labels_bool]
    neg_probs = probs[~labels_bool]
    return {
        "n": int(len(labels)),
        "positives": int(labels_bool.sum()),
        "prevalence": float(labels_bool.mean()),
        "auroc": float(roc_auc_score(labels_bool, probs)) if labels_bool.nunique() == 2 else math.nan,
        "average_precision": float(average_precision_score(labels_bool, probs))
        if labels_bool.nunique() == 2
        else math.nan,
        "brier": float(brier_score_loss(labels_bool, probs)),
        "accuracy_at_0_5": (tp + tn) / len(labels) if len(labels) else math.nan,
        "sensitivity_at_0_5": tp / (tp + fn) if (tp + fn) else math.nan,
        "specificity_at_0_5": tn / (tn + fp) if (tn + fp) else math.nan,
        "ppv_at_0_5": tp / (tp + fp) if (tp + fp) else math.nan,
        "npv_at_0_5": tn / (tn + fn) if (tn + fn) else math.nan,
        "mean_prob_positive": float(pos_probs.mean()) if len(pos_probs) else math.nan,
        "mean_prob_negative": float(neg_probs.mean()) if len(neg_probs) else math.nan,
    }


def run_outcome(frame, outcome: str, min_control_followup_days: int, args, numeric_cols, categorical_cols):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    from sklearn.utils.class_weight import compute_sample_weight

    label = frame[outcome].astype(int)
    eligible = frame["aki_any"] | frame["followup_days"].ge(min_control_followup_days)
    data = frame.loc[eligible].copy()
    y = data[outcome].astype(int)
    X = data[numeric_cols + categorical_cols].copy()

    min_class = int(y.value_counts().min())
    n_splits = min(args.n_splits, min_class)
    if n_splits < 2:
        raise ValueError(f"Not enough class balance for {outcome}: {y.value_counts().to_dict()}")

    oof = np.zeros(len(data), dtype=float)
    fold_rows = []
    used_model_type = None
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        used_model_type, model = choose_model(args.model_type, args.seed + fold, y_train)
        pipe = make_pipeline(numeric_cols, categorical_cols, model)
        fit_kwargs = {}
        if args.class_weight == "balanced":
            fit_kwargs["model__sample_weight"] = compute_sample_weight(
                class_weight="balanced",
                y=y_train,
            )
        pipe.fit(X_train, y_train, **fit_kwargs)
        probs = pipe.predict_proba(X_test)[:, 1]
        oof[test_idx] = probs
        fold_metric = compute_metrics(y_test.reset_index(drop=True), pd.Series(probs))
        fold_metric.update({"outcome": outcome, "fold": fold, "model_type": used_model_type})
        fold_rows.append(fold_metric)

    overall = compute_metrics(y.reset_index(drop=True), pd.Series(oof))
    overall.update(
        {
            "outcome": outcome,
            "fold": "overall",
            "model_type": used_model_type,
            "eligible_n": int(len(data)),
            "feature_set": "demo_charlson_regimen" if args.include_regimen else "demo_charlson",
        }
    )

    prediction_rows = pd.DataFrame(
        {
            "person_id": data["person_id"].to_numpy(),
            "outcome": outcome,
            "label": y.to_numpy(),
            "oof_probability": oof,
        }
    )
    return overall, fold_rows, prediction_rows


def main() -> None:
    import pandas as pd

    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_analysis_frame(args.denominator_csv, args.baseline_features_csv)
    numeric_cols, categorical_cols = feature_columns(frame, args.include_regimen)
    if not numeric_cols and not categorical_cols:
        raise ValueError("No usable feature columns found.")

    summary_rows = []
    fold_rows = []
    pred_frames = []
    for outcome, default_min_followup in OUTCOMES:
        min_followup = args.min_any_control_followup_days if outcome == "aki_any" else default_min_followup
        overall, folds, preds = run_outcome(
            frame,
            outcome,
            min_followup,
            args,
            numeric_cols,
            categorical_cols,
        )
        summary_rows.append(overall)
        fold_rows.extend(folds)
        pred_frames.append(preds)

    summary = pd.DataFrame(summary_rows)
    folds = pd.DataFrame(fold_rows)
    predictions = pd.concat(pred_frames, ignore_index=True)

    summary_path = out_dir / "aki_ml_baseline_summary.csv"
    folds_path = out_dir / "aki_ml_baseline_folds.csv"
    preds_path = out_dir / "aki_ml_baseline_oof_predictions.csv"
    manifest_path = out_dir / "aki_ml_baseline_manifest.json"
    summary.to_csv(summary_path, index=False)
    folds.to_csv(folds_path, index=False)
    predictions.to_csv(preds_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "denominator_csv": str(args.denominator_csv),
                "baseline_features_csv": str(args.baseline_features_csv),
                "feature_set": "demo_charlson_regimen" if args.include_regimen else "demo_charlson",
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "model_type_requested": args.model_type,
                "class_weight": args.class_weight,
                "seed": args.seed,
                "n_splits_requested": args.n_splits,
                "n_rows_merged": int(len(frame)),
            },
            indent=2,
            sort_keys=True,
        )
    )

    print("AKI_ML_BASELINE_SUMMARY")
    cols = [
        "outcome",
        "model_type",
        "n",
        "positives",
        "prevalence",
        "auroc",
        "average_precision",
        "brier",
        "accuracy_at_0_5",
        "sensitivity_at_0_5",
        "specificity_at_0_5",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
