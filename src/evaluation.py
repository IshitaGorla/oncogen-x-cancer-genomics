"""Patient-held-out static/temporal benchmark for fully observed state tables."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             average_precision_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from src.temporal_features import temporal_features, feature_columns


def run_benchmark(states: pd.DataFrame, gene_columns: list[str], output: str | Path,
                  seed: int = 42) -> pd.DataFrame:
    """Benchmark four configurations with one patient-grouped split.

    Only genes explicitly represented as complete observed 0/1 states are valid.
    No metric is computed for a target with one class in the test set.
    """
    out = Path(output)
    for folder in ("metrics", "predictions", "models"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    frame = temporal_features(states, gene_columns)
    if frame.empty or frame.patient_id.nunique() < 4:
        raise ValueError("At least four independent patients with consecutive states are required.")
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_i, test_i = next(split.split(frame, groups=frame.patient_id))
    train, test = frame.iloc[train_i], frame.iloc[test_i]
    summaries, predictions, importances = [], [], []
    for representation, temporal in (("static", False), ("temporal", True)):
        cols = feature_columns(frame, temporal)
        X_train, X_test = train[cols], test[cols]
        for model_name in ("logistic_regression", "random_forest"):
            for gene in gene_columns:
                target = f"target__{gene}"
                y_train, y_test = train[target].astype(int), test[target].astype(int)
                if y_train.nunique() < 2:
                    summaries.append(_metric_row(representation, model_name, gene, y_test, None, None,
                                                  frame, train, test, "training target has one class"))
                    continue
                estimator = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000,
                             class_weight="balanced", random_state=seed)) if model_name == "logistic_regression"
                             else RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                 class_weight="balanced_subsample", random_state=seed, n_jobs=-1))
                estimator.fit(X_train, y_train)
                prob = estimator.predict_proba(X_test)[:, 1]
                pred = (prob >= 0.5).astype(int)
                if model_name == "random_forest":
                    for feature, importance in zip(cols, estimator.feature_importances_):
                        importances.append({"representation": representation, "gene": gene,
                                            "feature": feature, "importance": float(importance)})
                summaries.append(_metric_row(representation, model_name, gene, y_test, pred, prob,
                                              frame, train, test, ""))
                for j, (_, row) in enumerate(test.iterrows()):
                    predictions.append({"representation": representation, "model": model_name,
                        "gene": gene, "patient_id": row.patient_id, "cutoff_sample": row.cutoff_sample,
                        "target_sample": row.target_sample, "observed": int(y_test.iloc[j]),
                        "predicted": int(pred[j]), "score": float(prob[j])})
                joblib.dump({"estimator": estimator, "feature_columns": cols, "gene": gene,
                             "representation": representation, "seed": seed},
                            out / "models" / f"{representation}_{model_name}_{gene}.joblib")
    metrics = pd.DataFrame(summaries)
    metric_rows = metrics.to_dict("records")
    for record in metric_rows:
        record["aggregation_level"] = "gene"
    for representation in ("static", "temporal"):
        for model_name in ("logistic_regression", "random_forest"):
            gene_rows = [r for r in metric_rows if r["representation"] == representation and
                         r["model"] == model_name and r.get("aggregation_level") == "gene"]
            if not gene_rows:
                continue
            usable = [r for r in gene_rows if r["f1"] is not None and not pd.isna(r["f1"])]
            macro = {"representation": representation, "model": model_name, "gene": "__macro__",
                     "aggregation_level": "macro", "patients_total": frame.patient_id.nunique(),
                     "train_patients": train.patient_id.nunique(), "test_patients": test.patient_id.nunique(),
                     "test_transitions": len(test), "test_positive": sum(r["test_positive"] for r in gene_rows),
                     "note": "Unweighted mean across evaluable genes."}
            for name in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
                values = [r[name] for r in gene_rows if r[name] is not None and not pd.isna(r[name])]
                macro[name] = sum(values) / len(values) if values else None
            for name in ("tn", "fp", "fn", "tp"):
                macro[name] = None
            metric_rows.append(macro)

            pred_rows = [r for r in predictions if r["representation"] == representation and r["model"] == model_name]
            if pred_rows:
                y_true = [r["observed"] for r in pred_rows]
                y_pred = [r["predicted"] for r in pred_rows]
                y_score = [r["score"] for r in pred_rows]
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
                micro = {"representation": representation, "model": model_name, "gene": "__micro__",
                         "aggregation_level": "micro", "patients_total": frame.patient_id.nunique(),
                         "train_patients": train.patient_id.nunique(), "test_patients": test.patient_id.nunique(),
                         "test_transitions": len(test), "test_positive": int(sum(y_true)),
                         "accuracy": accuracy_score(y_true, y_pred),
                         "precision": precision_score(y_true, y_pred, zero_division=0),
                         "recall": recall_score(y_true, y_pred, zero_division=0),
                         "f1": f1_score(y_true, y_pred, zero_division=0), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                         "note": "Micro metrics pool held-out gene-transition labels."}
                micro["roc_auc"] = roc_auc_score(y_true, y_score) if len(set(y_true)) == 2 else None
                micro["pr_auc"] = average_precision_score(y_true, y_score) if len(set(y_true)) == 2 else None
                metric_rows.append(micro)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "metrics" / "model_comparison.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "predictions" / "heldout_predictions.csv", index=False)
    pd.DataFrame(importances).to_csv(out / "metrics" / "random_forest_feature_importance.csv", index=False)
    metadata = {"seed": seed, "split": "single GroupShuffleSplit, 25% held-out patients",
                "train_patients": sorted(train.patient_id.unique().tolist()),
                "test_patients": sorted(test.patient_id.unique().tolist()),
                "train_transitions": len(train), "test_transitions": len(test),
                "genes": gene_columns, "target": "gene binary status at next observed sample",
                "caveat": "Interpret as alteration-state forecasting only when assay coverage and time order are verified."}
    (out / "metrics" / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics


def _metric_row(rep, model, gene, truth, pred, probability, frame, train, test, note):
    auc = ap = None
    if pred is None:
        acc = precision = recall = f1 = None
        tn = fp = fn = tp = None
    else:
        acc = accuracy_score(truth, pred)
        precision = precision_score(truth, pred, zero_division=0)
        recall = recall_score(truth, pred, zero_division=0)
        f1 = f1_score(truth, pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel().tolist()
        if truth.nunique() > 1:
            auc = roc_auc_score(truth, probability)
            ap = average_precision_score(truth, probability)
    return {"representation": rep, "model": model, "gene": gene, "aggregation_level": "gene",
            "patients_total": frame.patient_id.nunique(), "train_patients": train.patient_id.nunique(),
            "test_patients": test.patient_id.nunique(), "test_transitions": len(test),
            "test_positive": int(truth.sum()), "accuracy": acc, "precision": precision,
            "recall": recall, "f1": f1, "roc_auc": auc, "pr_auc": ap,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp, "note": note or
            ("ROC-AUC/PR-AUC undefined: test target has one class." if truth.nunique() < 2 else "")}
