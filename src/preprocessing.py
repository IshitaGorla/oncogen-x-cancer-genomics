"""Small, transparent preprocessing helpers for exploration."""
import pandas as pd


def filter_genomic_data(df: pd.DataFrame, patients=None, cancer_types=None, timepoints=None, query=""):
    """Apply exploration filters and a case-insensitive search across displayed fields."""
    result = df.copy()
    if patients:
        result = result[result["patient_id"].isin(patients)]
    if cancer_types:
        result = result[result["cancer_type"].isin(cancer_types)]
    if timepoints:
        result = result[result["timepoint"].isin(timepoints)]
    if query:
        searchable = result.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False))
        result = result[searchable.any(axis=1)]
    return result.reset_index(drop=True)


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise missingness for the active dataset."""
    counts = df.isna().sum()
    return pd.DataFrame({"Column": counts.index, "Missing values": counts.values,
                         "Missing (%)": (counts.values / max(len(df), 1) * 100).round(1)})
