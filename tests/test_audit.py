"""Optional-ID audit behavior and compatibility checks."""

import unittest

import pandas as pd

from ml_gearbox.audit import duplicate_report, id_report, run_data_audit


class OptionalIdAuditTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {"value": [1, 1, 2, 3], "target": [0, 0, 1, 1]},
            index=[9, 9, 4, 2],
        )

    def test_omitted_id_detects_duplicate_observations_without_mutating_input(self):
        original = self.df.copy(deep=True)

        reports = run_data_audit(self.df, target_col="target")

        self.assertIsNone(reports["id"])
        self.assertEqual(reports["duplicates"], {"full_row_duplicates": 1})
        self.assertEqual(reports["target"]["count"].tolist(), [2, 2])
        self.assertEqual(reports["schema"].index.tolist(), ["value", "target"])
        pd.testing.assert_frame_equal(self.df, original)

    def test_omitted_id_does_not_infer_named_identifier_columns(self):
        df = self.df.assign(id=["a", "a", "b", "c"])

        reports = run_data_audit(df, target_col="target")

        self.assertIsNone(reports["id"])
        self.assertEqual(reports["duplicates"], {"full_row_duplicates": 1})

    def test_explicit_none_is_equivalent_to_omitting_id(self):
        reports = run_data_audit(self.df, target_col="target", id_col=None)

        self.assertIsNone(reports["id"])
        self.assertEqual(reports["duplicates"], {"full_row_duplicates": 1})

    def test_existing_id_retains_identifier_and_duplicate_report_output(self):
        df = pd.DataFrame(
            {
                "id": ["a", "a", "b", "b", None],
                "value": [1, 1, 2, 3, 4],
                "target": [0, 0, 1, 1, 0],
            }
        )
        original = df.copy(deep=True)

        # Preserve the original third positional argument as well.
        reports = run_data_audit(df, "target", "id")

        self.assertEqual(
            reports["id"],
            {"total_rows": 5, "unique_ids": 2, "missing_ids": 1, "duplicate_ids": 2},
        )
        self.assertEqual(
            reports["duplicates"],
            {
                "full_row_duplicates": 1,
                "duplicate_id_rows": 2,
                "conflicting_id_rows": 2,
            },
        )
        self.assertEqual(reports["id"], id_report(df, "id"))
        self.assertEqual(reports["duplicates"], duplicate_report(df, "id"))
        pd.testing.assert_frame_equal(df, original)

    def test_other_reports_and_configured_checks_remain_available_without_id(self):
        df = self.df.assign(id=["a", "a", "b", "c"])
        options = {
            "expected_schema": {"value": "int64", "target": "int64", "id": "str"},
            "range_rules": {"value": {"min": 0}},
            "cross_field_rules": [("value", ">=", "target")],
        }
        with_id = run_data_audit(df, target_col="target", id_col="id", **options)
        without_id = run_data_audit(df, target_col="target", **options)

        self.assertEqual(set(with_id), set(without_id))
        for name in with_id.keys() - {"id", "duplicates"}:
            with self.subTest(report=name):
                pd.testing.assert_frame_equal(with_id[name], without_id[name])
        self.assertIsNone(without_id["id"])

    def test_missing_explicit_id_is_an_error_instead_of_an_omitted_id(self):
        with self.assertRaisesRegex(ValueError, "id_col 'missing_id'.*dataframe"):
            run_data_audit(self.df, target_col="target", id_col="missing_id")

    def test_missing_target_has_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "target_col 'missing_target'.*dataframe"):
            run_data_audit(self.df, target_col="missing_target")

    def test_invalid_explicit_id_names_are_rejected(self):
        for value, error in [("", ValueError), (42, TypeError), (["id"], TypeError)]:
            with self.subTest(id_col=value):
                with self.assertRaisesRegex(error, "id_col must be a non-empty string"):
                    run_data_audit(self.df, target_col="target", id_col=value)

    def test_standalone_id_checks_require_a_valid_existing_id(self):
        for check in [id_report, duplicate_report]:
            with self.subTest(check=check.__name__):
                with self.assertRaisesRegex(ValueError, "id_col 'missing_id'.*dataframe"):
                    check(self.df, "missing_id")
        with self.assertRaisesRegex(TypeError, "id_col must be a non-empty string"):
            id_report(self.df, None)


if __name__ == "__main__":
    unittest.main()
