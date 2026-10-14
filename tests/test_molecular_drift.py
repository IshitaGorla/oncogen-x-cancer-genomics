import unittest

import pandas as pd

from src.molecular_drift import analyze_patient, compare_demo_rules, make_demo_data


def patient_frame(values):
    return pd.DataFrame({
        "patient_id": "fictional-p1",
        "observation": range(1, len(values) + 1),
        "signal_units": values,
    })


class MolecularDriftTests(unittest.TestCase):
    def test_baseline_uses_first_valid_measurements_and_computes_deviation(self):
        result = analyze_patient(patient_frame([10.0, 12.0, 14.0, 14.0]), "fictional-p1", threshold=1, persistence=2, baseline_n=2)
        self.assertEqual(result["baseline"], 11.0)
        self.assertEqual(result["baseline_observations"], [1, 2])
        self.assertEqual(result["observations"].deviation.tolist(), [-1.0, 1.0, 3.0, 3.0])

    def test_persistence_requires_consecutive_same_direction_observations(self):
        result = analyze_patient(patient_frame([10, 10, 12, 12]), "fictional-p1", threshold=1, persistence=2, baseline_n=2)
        self.assertEqual(result["current_persistence"], 2)
        self.assertEqual(result["state"], "persistent change for research review")

    def test_missing_observation_breaks_persistence_and_is_counted(self):
        result = analyze_patient(patient_frame([10, 10, 12, None, 12]), "fictional-p1", threshold=1, persistence=2, baseline_n=2)
        self.assertEqual(result["missing_count"], 1)
        self.assertEqual(result["current_persistence"], 1)
        self.assertEqual(result["state"], "isolated change")

    def test_latest_missing_measurement_returns_insufficient_data(self):
        result = analyze_patient(patient_frame([10, 10, 12, 12, None]), "fictional-p1", threshold=1, persistence=2, baseline_n=2)
        self.assertFalse(result["sufficient"])
        self.assertEqual(result["state"], "insufficient data")

    def test_insufficient_valid_observations_returns_insufficient_data(self):
        result = analyze_patient(patient_frame([10, 10, 12]), "fictional-p1", threshold=1, persistence=2, baseline_n=2)
        self.assertEqual(result["state"], "insufficient data")

    def test_nonpersistent_adequate_trajectory_returns_isolated_state(self):
        result = analyze_patient(patient_frame([10, 10, 12, 10, 10]), "fictional-p1", threshold=1, persistence=2, baseline_n=2)
        self.assertEqual(result["state"], "isolated change")
        self.assertEqual(result["current_persistence"], 0)

    def test_invalid_threshold_and_unknown_patient_are_rejected(self):
        data = patient_frame([10, 10, 12, 12])
        with self.assertRaisesRegex(ValueError, "threshold"):
            analyze_patient(data, "fictional-p1", threshold=0)
        with self.assertRaisesRegex(ValueError, "Unknown patient"):
            analyze_patient(data, "unknown")

    def test_nonnumeric_measurements_are_not_silently_treated_as_missing(self):
        data = patient_frame([10, 10, 12, 12])
        data["signal_units"] = data.signal_units.astype(object)
        data.loc[2, "signal_units"] = "not-a-number"
        with self.assertRaisesRegex(ValueError, "numeric values"):
            analyze_patient(data, "fictional-p1")

    def test_demo_is_deterministic_and_comparison_uses_planted_synthetic_labels(self):
        data_a, truth_a = make_demo_data()
        data_b, truth_b = make_demo_data()
        pd.testing.assert_frame_equal(data_a, data_b)
        self.assertEqual(truth_a, truth_b)
        metrics, outcomes = compare_demo_rules(data_a, truth_a)
        self.assertEqual(set(metrics.rule), {"single measurement", "trajectory rule"})
        self.assertTrue((metrics.patients == len(truth_a)).all())
        self.assertEqual(len(outcomes), 2 * len(truth_a))


if __name__ == "__main__":
    unittest.main()
