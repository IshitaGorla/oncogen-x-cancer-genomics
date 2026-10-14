import pandas as pd
import unittest

from src.temporal_features import temporal_features, validate_states
from src.data_loader import load_genomic_data, validate_genomic_data


def fixture_states():
    return pd.DataFrame([
        {"patient_id": "p1", "sample_id": "s1", "timepoint": 1, "TP53": 0, "KRAS": 0},
        {"patient_id": "p1", "sample_id": "s2", "timepoint": 2, "TP53": 1, "KRAS": 0},
        {"patient_id": "p1", "sample_id": "s3", "timepoint": 3, "TP53": 1, "KRAS": 1},
        {"patient_id": "p2", "sample_id": "s1", "timepoint": 1, "TP53": 0, "KRAS": 1},
        {"patient_id": "p2", "sample_id": "s2", "timepoint": 2, "TP53": 0, "KRAS": 1},
])


class TemporalFeatureTests(unittest.TestCase):
    def test_temporal_features_only_include_past_and_targets_are_next_state(self):
        result = temporal_features(fixture_states(), ["TP53", "KRAS"])
        p1_second = result[(result.patient_id == "p1") & (result.cutoff_sample == "s2")].iloc[0]
        self.assertEqual(p1_second["TP53__current"], 1)
        self.assertEqual(p1_second["TP53__historical_frequency"], 0.5)
        self.assertEqual(p1_second["TP53__last_acquired_steps"], 0)
        self.assertEqual(p1_second["target__KRAS"], 1)
        from src.temporal_features import feature_columns
        self.assertFalse(any(c.startswith("target__") for c in feature_columns(result, True)))


    def test_missing_state_is_not_assumed_negative(self):
        states = fixture_states()
        states.loc[0, "TP53"] = float("nan")
        with self.assertRaisesRegex(ValueError, "missing is not negative"):
            validate_states(states, ["TP53", "KRAS"])


    def test_duplicate_patient_sample_is_rejected(self):
        states = fixture_states()
        states.loc[1, "sample_id"] = "s1"
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_states(states, ["TP53", "KRAS"])

    def test_blank_identifiers_are_rejected(self):
        states = fixture_states()
        states.loc[0, "patient_id"] = "  "
        with self.assertRaisesRegex(ValueError, "non-empty"):
            validate_states(states, ["TP53", "KRAS"])

    def test_co_temporal_samples_are_rejected(self):
        states = fixture_states()
        states.loc[1, "timepoint"] = 1
        with self.assertRaisesRegex(ValueError, "unique timepoint"):
            temporal_features(states, ["TP53", "KRAS"])

    def test_unordered_labels_and_invalid_dates_are_rejected(self):
        states = fixture_states().query("patient_id == 'p1'").copy()
        states["timepoint"] = ["T1", "T10", "T2"]
        with self.assertRaisesRegex(ValueError, "numeric values or valid ISO"):
            temporal_features(states, ["TP53", "KRAS"])
        states["timepoint"] = ["2020-01-01", "2020-02-30", "2020-03-01"]
        with self.assertRaisesRegex(ValueError, "invalid ISO"):
            temporal_features(states, ["TP53", "KRAS"])

    def test_numeric_string_timepoints_sort_numerically(self):
        states = fixture_states().query("patient_id == 'p1'").copy()
        states["timepoint"] = ["10", "2", "1"]
        result = temporal_features(states, ["TP53", "KRAS"])
        self.assertEqual(result.iloc[0]["cutoff_sample"], "s3")
        self.assertEqual(result.iloc[0]["target_sample"], "s2")

    def test_real_iso_dates_produce_observed_gap_feature(self):
        states = fixture_states().query("patient_id == 'p1'").copy()
        states["timepoint"] = ["2020-01-01", "2020-01-08", "2020-01-18"]
        result = temporal_features(states, ["TP53", "KRAS"])
        self.assertEqual(result.iloc[0]["days_since_previous_sample"], -1.0)
        self.assertEqual(result.iloc[1]["days_since_previous_sample"], 7.0)


    def test_provided_synthetic_demo_is_validated_and_labeled(self):
        data = load_genomic_data("data/sample/demo_genomic_data.csv")
        self.assertFalse(validate_genomic_data(data))
        self.assertTrue(data.data_source.str.contains("synthetic", case=False).all())

    def test_cleaning_preserves_missingness(self):
        from src.data_loader import clean_genomic_data
        frame = pd.DataFrame({"patient_id": [" p1 ", None]})
        cleaned = clean_genomic_data(frame)
        self.assertEqual(cleaned.patient_id.iloc[0], "p1")
        self.assertTrue(pd.isna(cleaned.patient_id.iloc[1]))

    def test_cbioportal_mutations_join_sample_patient_without_time_inference(self):
        from src.download_cbioportal import _mutation_events
        events = _mutation_events([{"sampleId": "S1", "gene": {"hugoGeneSymbol": "TP53"},
                                    "proteinChange": "p.R1C", "mutationType": "Missense"}], {"S1": "P1"})
        self.assertEqual(events.loc[0, "patient_id"], "P1")
        self.assertEqual(events.loc[0, "timepoint"], "single cross-sectional sample")
        self.assertEqual(events.loc[0, "gene"], "TP53")


if __name__ == "__main__":
    unittest.main()
