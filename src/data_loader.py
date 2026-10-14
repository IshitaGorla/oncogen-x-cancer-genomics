"""Validated data ingestion for the ONCOGEN-X research prototype."""
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "patient_id", "cancer_type", "timepoint", "gene", "alteration",
    "alteration_type", "genomic_status", "data_source",
]


def validate_genomic_data(df: pd.DataFrame) -> list[str]:
    """Return a list of required columns absent from a candidate dataset."""
    return [column for column in REQUIRED_COLUMNS if column not in df.columns]


def clean_genomic_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise strings and handle basic missing values without inventing data."""
    cleaned = df.copy()
    for column in cleaned.columns:
        if pd.api.types.is_string_dtype(cleaned[column].dtype) or cleaned[column].dtype == "object":
            # Normalize text while preserving nulls for honest missingness reporting.
            cleaned[column] = cleaned[column].map(lambda value: value.strip() if isinstance(value, str) else value)
    return cleaned.drop_duplicates().reset_index(drop=True)


def load_genomic_data(source: str | Path | object) -> pd.DataFrame:
    """Read CSV data, validate its schema, and return a clean DataFrame."""
    data = pd.read_csv(source)
    missing = validate_genomic_data(data)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    return clean_genomic_data(data)
