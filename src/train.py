"""Run the patient-held-out static vs temporal benchmark."""
import argparse
from pathlib import Path
import pandas as pd
from src.evaluation import run_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("states", help="CSV: patient_id,sample_id,timepoint plus complete binary gene columns")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    states = pd.read_csv(args.states)
    meta = {"patient_id", "sample_id", "timepoint"}
    genes = [c for c in states.columns if c not in meta]
    if not genes:
        parser.error("No gene-state columns found.")
    results = run_benchmark(states, genes, Path(args.output), args.seed)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
