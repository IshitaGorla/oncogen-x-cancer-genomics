"""Fictional, deterministic data and transparent rules for a drift demo page."""

import numpy as np
import pandas as pd


def make_demo_data(seed=42):
    """Create fictional measurements and planted simulation labels (not clinical data)."""
    rng = np.random.default_rng(seed)
    rows = []
    truth = {}
    for patient_number in range(1, 13):
        patient_id = f"DRIFT-DEMO-{patient_number:02d}"
        scenario = "persistent" if patient_number <= 6 else "isolated" if patient_number <= 9 else "stable"
        truth[patient_id] = int(scenario == "persistent")
        personal_level = 8.0 + patient_number * 0.35
        for observation in range(1, 9):
            value = personal_level + rng.normal(0, 0.12)
            if scenario == "persistent" and observation >= 4:
                value += 2.2 + rng.normal(0, 0.12)
            elif scenario == "isolated" and observation == 5:
                value += 2.2
            if (patient_id, observation) in {
                ("DRIFT-DEMO-02", 6), ("DRIFT-DEMO-08", 7), ("DRIFT-DEMO-11", 3)
            }:
                value = np.nan
            rows.append({
                "patient_id": patient_id,
                "observation": observation,
                "signal_units": round(float(value), 3) if not np.isnan(value) else np.nan,
            })
    return pd.DataFrame(rows), truth


def analyze_patient(data, patient_id, threshold=1.0, persistence=2, baseline_n=2):
    """Calculate baseline-relative deviations and a trailing same-direction rule state."""
    required = {"patient_id", "observation", "signal_units"}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be a finite positive number")
    if not isinstance(persistence, int) or persistence < 1:
        raise ValueError("persistence must be a positive integer")
    if not isinstance(baseline_n, int) or baseline_n < 1:
        raise ValueError("baseline_n must be a positive integer")

    patient = data.loc[data.patient_id == patient_id].copy()
    if patient.empty:
        raise ValueError(f"Unknown patient_id: {patient_id}")
    raw_observation = patient.observation.copy()
    raw_signal = patient.signal_units.copy()
    patient["observation"] = pd.to_numeric(raw_observation, errors="coerce")
    patient["signal_units"] = pd.to_numeric(raw_signal, errors="coerce")
    if (raw_observation.notna() & patient.observation.isna()).any():
        raise ValueError("observation values must be numeric")
    if (raw_signal.notna() & patient.signal_units.isna()).any():
        raise ValueError("signal_units must contain numeric values or explicit missing values")
    if patient.observation.isna().any() or patient.observation.duplicated().any():
        raise ValueError("observation values must be numeric and unique within patient")
    if patient.signal_units.dropna().map(np.isfinite).eq(False).any():
        raise ValueError("signal_units must contain finite numbers or missing values")
    patient = patient.sort_values("observation").reset_index(drop=True)
    patient["deviation"] = np.nan
    patient["baseline_observation"] = False
    valid_indices = patient.index[patient.signal_units.notna()].tolist()
    enough_baseline = len(valid_indices) >= baseline_n
    baseline_indices = valid_indices[:baseline_n] if enough_baseline else valid_indices
    patient.loc[baseline_indices, "baseline_observation"] = True
    baseline = float(patient.loc[baseline_indices, "signal_units"].mean()) if enough_baseline else np.nan
    if enough_baseline:
        patient["deviation"] = patient.signal_units - baseline

    patient["beyond_threshold"] = patient.deviation.abs() >= threshold
    patient["consecutive_deviations"] = 0
    streak = 0
    prior_direction = 0
    for index, row in patient.iterrows():
        if row.baseline_observation:
            continue
        if pd.isna(row.signal_units) or not row.beyond_threshold:
            streak, prior_direction = 0, 0
        else:
            direction = 1 if row.deviation > 0 else -1
            streak = streak + 1 if direction == prior_direction else 1
            prior_direction = direction
        patient.at[index, "consecutive_deviations"] = streak

    missing_count = int(patient.signal_units.isna().sum())
    sufficient = enough_baseline and len(valid_indices) >= baseline_n + persistence and pd.notna(patient.signal_units.iloc[-1])
    final_streak = int(patient.consecutive_deviations.iloc[-1]) if len(patient) else 0
    if not sufficient:
        state = "insufficient data"
    elif final_streak >= persistence:
        state = "persistent change for research review"
    else:
        state = "isolated change"

    return {
        "observations": patient,
        "baseline": baseline,
        "baseline_observations": patient.loc[patient.baseline_observation, "observation"].tolist(),
        "missing_count": missing_count,
        "valid_count": len(valid_indices),
        "sufficient": bool(sufficient),
        "current_persistence": final_streak,
        "state": state,
        "threshold": float(threshold),
        "persistence_required": persistence,
        "baseline_n": baseline_n,
    }


def compare_demo_rules(data, truth, threshold=1.0, persistence=2, baseline_n=2):
    """Compare two toy rules with planted synthetic patient labels; return counts and metrics."""
    rows = []
    for patient_id, expected in truth.items():
        result = analyze_patient(data, patient_id, threshold, persistence, baseline_n)
        post_baseline = result["observations"].loc[~result["observations"].baseline_observation]
        single_measurement = bool((post_baseline.deviation.abs() >= threshold).any())
        trajectory_rule = result["state"] == "persistent change for research review"
        for rule, predicted in [("single measurement", single_measurement), ("trajectory rule", trajectory_rule)]:
            rows.append({"patient_id": patient_id, "rule": rule, "truth": int(expected), "predicted": int(predicted)})
    outcomes = pd.DataFrame(rows)
    metrics = []
    for rule, subset in outcomes.groupby("rule", sort=False):
        tp = int(((subset.truth == 1) & (subset.predicted == 1)).sum())
        fp = int(((subset.truth == 0) & (subset.predicted == 1)).sum())
        fn = int(((subset.truth == 1) & (subset.predicted == 0)).sum())
        tn = int(((subset.truth == 0) & (subset.predicted == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        metrics.append({
            "rule": rule, "patients": len(subset), "accuracy": (tp + tn) / len(subset),
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })
    return pd.DataFrame(metrics), outcomes
