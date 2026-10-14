import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation import run_benchmark


class GroupedBenchmarkTests(unittest.TestCase):
    def test_metrics_and_predictions_are_written_with_patient_separation(self):
        rows = []
        for patient in range(16):
            for timepoint in range(3):
                rows.append({"patient_id": f"p{patient}", "sample_id": f"s{patient}_{timepoint}",
                             "timepoint": timepoint,
                             "TP53": int((patient // 2 + timepoint) % 4 >= 2),
                             "KRAS": int((patient + timepoint) % 3 == 0)})
        states = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory(dir="outputs") as directory:
            metrics = run_benchmark(states, ["TP53", "KRAS"], directory, seed=42)
            metadata = json.loads((Path(directory) / "metrics" / "run_metadata.json").read_text())
            self.assertTrue((Path(directory) / "predictions" / "heldout_predictions.csv").exists())
            self.assertTrue((Path(directory) / "metrics" / "random_forest_feature_importance.csv").exists())
            self.assertTrue(set(metadata["train_patients"]).isdisjoint(metadata["test_patients"]))
            self.assertEqual(set(metrics["representation"]), {"static", "temporal"})
            fitted = metrics[metrics["aggregation_level"] == "gene"]
            self.assertEqual(set(zip(fitted["representation"], fitted["model"])), {
                ("static", "logistic_regression"), ("static", "random_forest"),
                ("temporal", "logistic_regression"), ("temporal", "random_forest")})
            predictions = pd.read_csv(Path(directory) / "predictions" / "heldout_predictions.csv")
            self.assertEqual(set(zip(predictions["representation"], predictions["model"])), {
                ("static", "logistic_regression"), ("static", "random_forest"),
                ("temporal", "logistic_regression"), ("temporal", "random_forest")})
            self.assertEqual(len(list((Path(directory) / "models").glob("*.joblib"))), 8)


if __name__ == "__main__":
    unittest.main()
