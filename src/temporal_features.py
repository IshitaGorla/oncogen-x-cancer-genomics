"""Leakage-safe temporal features for explicitly observed patient states.

Input states are wide tables with one row per patient/sample and binary gene
columns. Only genes measured at each sample may be used; callers must exclude
unmeasured states rather than treating missing values as zero.
"""
from __future__ import annotations

import pandas as pd

META = ["patient_id", "sample_id", "timepoint"]


def _ordered_timepoints(values: pd.Series) -> tuple[pd.Series, bool]:
    """Normalize supported timepoints; reject labels without temporal semantics."""
    if pd.api.types.is_numeric_dtype(values.dtype) and not pd.api.types.is_bool_dtype(values.dtype):
        order = pd.to_numeric(values, errors="coerce")
        if not order.map(lambda x: pd.notna(x) and abs(x) != float("inf")).all():
            raise ValueError("Numeric timepoints must be finite.")
        return order, False
    strings = values.astype(str)
    numeric = pd.to_numeric(strings, errors="coerce")
    if numeric.notna().all():
        if not numeric.map(lambda x: pd.notna(x) and abs(x) != float("inf")).all():
            raise ValueError("Numeric timepoints must be finite.")
        return numeric, False
    # Require ISO calendar dates/datetimes. Arbitrary labels (including T1/T2)
    # cannot establish chronology and must not be sorted into transitions.
    if not strings.str.match(r"^\d{4}-\d{2}-\d{2}(?:[T ][0-9:.+Zz-]+)?$").all():
        raise ValueError("Timepoints must be finite numeric values or valid ISO dates/datetimes.")
    parsed = pd.to_datetime(strings, errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError("Timepoints contain an invalid ISO date/datetime.")
    return parsed, True


def validate_states(states: pd.DataFrame, gene_columns: list[str]) -> None:
    missing = [c for c in META + gene_columns if c not in states.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    if not gene_columns or len(set(gene_columns)) != len(gene_columns) or set(gene_columns) & set(META):
        raise ValueError("Gene columns must be non-empty, unique, and separate from metadata.")
    if not states.columns.is_unique:
        raise ValueError("Input column names must be unique.")
    if states[META].isna().any().any():
        raise ValueError("Patient, sample and timepoint must be present.")
    if states[["patient_id", "sample_id"]].astype(str).apply(lambda col: col.str.strip().eq("")).any().any():
        raise ValueError("Patient and sample identifiers must be non-empty.")
    if states.duplicated(["patient_id", "sample_id"]).any():
        raise ValueError("Each patient/sample pair must be unique.")
    time_order, _ = _ordered_timepoints(states["timepoint"])
    temporal_keys = pd.DataFrame({"patient_id": states["patient_id"], "_time_order": time_order})
    if temporal_keys.duplicated(["patient_id", "_time_order"]).any():
        raise ValueError("Each patient must have a unique timepoint; co-temporal samples cannot define transitions.")
    values = states[gene_columns]
    if values.isna().any().any() or not values.isin([0, 1, False, True]).all().all():
        raise ValueError("Gene states must be observed binary values; missing is not negative.")


def temporal_features(states: pd.DataFrame, gene_columns: list[str]) -> pd.DataFrame:
    """Return one cutoff row per consecutive observed sample transition.

    `timepoint` must be sortable in true temporal order (dates or ordered values).
    History features only use samples at or before the cutoff; next state is
    intentionally not returned as a feature.
    """
    validate_states(states, gene_columns)
    time_order, is_iso_date = _ordered_timepoints(states["timepoint"])
    ordered = states.assign(_time_order=time_order)
    if is_iso_date:
        ordered = ordered.assign(_parsed_time=time_order)
    ordered = ordered.sort_values(["patient_id", "_time_order", "sample_id"], kind="stable")
    rows = []
    for patient, group in ordered.groupby("patient_id", sort=False):
        records = list(group.to_dict("records"))
        for idx in range(len(records) - 1):
            history = records[: idx + 1]
            current, future = history[-1], records[idx + 1]
            row = {"patient_id": patient, "cutoff_sample": current["sample_id"],
                   "target_sample": future["sample_id"], "cutoff_timepoint": current["timepoint"],
                   "target_timepoint": future["timepoint"],
                   "current_alteration_count": int(sum(int(current[g]) for g in gene_columns)),
                   "new_acquisition_count": int(sum(int(current[g]) and not int(history[-2][g])
                                                     for g in gene_columns)) if len(history) > 1 else 0,
                   "loss_count_since_previous": int(sum(not int(current[g]) and int(history[-2][g])
                                                         for g in gene_columns)) if len(history) > 1 else 0}
            if is_iso_date:
                row["days_since_previous_sample"] = (
                    float((current["_parsed_time"] - history[-2]["_parsed_time"]).total_seconds() / 86400)
                    if len(history) > 1 else -1.0)
            row["observations_so_far"] = len(history)
            for gene in gene_columns:
                vals = [int(r[gene]) for r in history]
                row[f"{gene}__current"] = vals[-1]
                row[f"{gene}__historical_frequency"] = sum(vals) / len(vals)
                row[f"{gene}__consecutive_persistence"] = next(
                    (n - 1 for n, value in enumerate(reversed(vals), start=1) if not value), len(vals))
                row[f"{gene}__ever_observed"] = int(any(vals))
                row[f"{gene}__last_acquired_steps"] = next(
                    (len(vals) - 1 - j for j in range(len(vals) - 1, 0, -1)
                     if vals[j] and not vals[j - 1]), -1)
                # Historical co-occurrence with each other gene, excluding current/future labels.
                for other in gene_columns:
                    if other != gene:
                        row[f"{gene}__cooccur_{other}"] = sum(int(r[gene]) * int(r[other]) for r in history) / len(history)
                row[f"target__{gene}"] = int(future[gene])
            rows.append(row)
    return pd.DataFrame(rows)


def feature_columns(frame: pd.DataFrame, temporal: bool) -> list[str]:
    excluded = {"patient_id", "cutoff_sample", "target_sample", "cutoff_timepoint", "target_timepoint"}
    excluded.update(c for c in frame.columns if c.startswith("target__"))
    if not temporal:
        return [c for c in frame.columns if c not in excluded and
                (c == "current_alteration_count" or c.endswith("__current"))]
    return [c for c in frame.columns if c not in excluded]
