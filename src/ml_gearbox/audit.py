import operator as op

import numpy as np
import pandas as pd

# ==========================================
# 1. TARGET, ID, DUPLICATES
# ==========================================


def target_report(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Returns the class distribution and percentages of the target variable."""
    summary = (
        df[target_col]
        .value_counts(dropna=False)
        .rename_axis("class")
        .reset_index(name="count")
    )
    summary["percentage"] = (summary["count"] / len(df)) * 100

    return summary


def id_report(df: pd.DataFrame, id_col: str) -> dict:
    """Audits the unique identifier column for missing or duplicate values."""
    return {
        "total_rows": len(df),
        "unique_ids": int(df[id_col].nunique()),
        "missing_ids": int(df[id_col].isna().sum()),
        "duplicate_ids": int(df[id_col].duplicated().sum()),
    }


def duplicate_report(df: pd.DataFrame, id_col: str | None = None) -> dict:
    """
    Audits the dataset for duplicate records at two levels:

    1. Full-row duplicates: rows that are identical across every column.
       These are almost always safe to drop (copy-paste / re-import errors).
    2. Conflicting ID rows (only if id_col is given): rows that share the
       same identifier but are NOT full-row duplicates, i.e. the same
       entity has two different sets of values on record. This is a more
       serious data-integrity problem than a plain duplicate, because you
       don't know which row is correct.
    """
    full_dupe_count = int(df.duplicated().sum())
    report = {"full_row_duplicates": full_dupe_count}

    if id_col is not None:
        full_dupe_mask = df.duplicated(keep=False)
        id_dupe_mask = df[id_col].duplicated(keep=False)
        conflicting_mask = id_dupe_mask & ~full_dupe_mask

        report["duplicate_id_rows"] = int(df[id_col].duplicated().sum())
        report["conflicting_id_rows"] = int(conflicting_mask.sum())

    return report


# ==========================================
# 2. MISSINGNESS
# ==========================================


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame of columns that contain missing values,
    including their counts and percentages.
    """

    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].reset_index()

    if missing_counts.empty:
        return pd.DataFrame(columns=["feature", "missing_count", "percentage"])

    missing_counts.columns = ["feature", "missing_count"]
    missing_counts["percentage"] = (missing_counts["missing_count"] / len(df)) * 100

    return missing_counts.sort_values("missing_count", ascending=False).reset_index(
        drop=True
    )


def hidden_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Detects blank strings or whitespace disguised as valid data in categorical columns."""
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    blank_summary = {}

    for column in categorical_cols:
        blanks = df[column].astype(str).str.strip().eq("").sum()
        if blanks > 0:
            blank_summary[column] = blanks

    if not blank_summary:
        return pd.DataFrame(columns=["feature", "hidden_missing_count", "percentage"])

    result = pd.Series(blank_summary, name="hidden_missing_count").to_frame()
    result["percentage"] = (result["hidden_missing_count"] / len(df)) * 100
    return (
        result.sort_values("hidden_missing_count", ascending=False)
        .reset_index(names="feature")
    )


def category_normalization_check(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Detects categories that would merge if stripped of whitespace and lowercased.

    This is a data-quality check, not an exploratory one: it flags columns
    where "USA", "usa", and " USA " are all being counted as different
    categories because of formatting rather than genuine differences.
    """
    records = []
    for col in columns:
        orig = df[col].nunique(dropna=False)
        norm = df[col].astype(str).str.strip().str.lower().nunique(dropna=False)

        if norm < orig:
            records.append(
                {"feature": col, "original_unique": orig, "normalized_unique": norm}
            )

    if not records:
        return pd.DataFrame(
            columns=["feature", "original_unique", "normalized_unique"]
        )

    return pd.DataFrame(records)


def infinite_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Checks all numeric columns for infinite values."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    infinite_counts = pd.Series(
        {column: np.isinf(df[column]).sum() for column in numeric_cols}
    )

    result = infinite_counts[infinite_counts > 0].reset_index()
    if result.empty:
        return pd.DataFrame(columns=["feature", "infinite_count"])

    result.columns = ["feature", "infinite_count"]
    return result


# ==========================================
# 3. SCHEMA, CARDINALITY, CONSTANTS
# ==========================================


def schema_report(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a master audit table summarizing column types, missingness, and uniqueness."""
    audit_table = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_count": df.isna().sum(),
            "missing_pct": df.isna().mean() * 100,
            "n_unique": df.nunique(dropna=False),
        }
    )
    audit_table["unique_pct"] = (audit_table["n_unique"] / len(df)) * 100
    return audit_table.sort_values("missing_pct", ascending=False)


def cardinality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Reports the unique value count for categorical features."""
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    return (
        pd.DataFrame(
            {
                "feature": categorical_cols,
                "n_unique": [df[col].nunique(dropna=False) for col in categorical_cols],
            }
        )
        .sort_values("n_unique", ascending=False)
        .reset_index(drop=True)
    )


def constant_feature_report(df: pd.DataFrame) -> pd.DataFrame:
    """Identifies constant features and near-constant (highly dominant) features."""
    dominant_pct = {
        col: df[col].value_counts(normalize=True, dropna=False).iloc[0] * 100
        for col in df.columns
    }

    report = pd.DataFrame(
        list(dominant_pct.items()), columns=["feature", "dominant_percentage"]
    )
    report["is_strictly_constant"] = report["dominant_percentage"] == 100.0

    return report.sort_values("dominant_percentage", ascending=False).reset_index(
        drop=True
    )


def numeric_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Enhances the standard describe() output with median, skew, and unique counts."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        return pd.DataFrame()

    summary = df[numeric_cols].describe().T
    summary["median"] = df[numeric_cols].median()
    summary["skew"] = df[numeric_cols].skew()
    summary["n_unique"] = df[numeric_cols].nunique()

    return summary.sort_values("skew", ascending=False)


# ==========================================
# 4. VALIDATION (requires a spec you define per dataset)
# ==========================================
#
# These three functions are deliberately configuration-driven rather than
# fully automatic: "is this valid?" is a question only YOU can answer for
# a given dataset (a negative age is invalid, a negative account balance
# might be perfectly normal). The checking logic below is generic and
# reusable; you supply the rules that are specific to the dataset at hand.


def schema_validation(df: pd.DataFrame, expected_schema: dict) -> pd.DataFrame:
    """
    Validates the dataframe's columns and dtypes against an expected schema.

    expected_schema: dict mapping column_name -> expected dtype, e.g.
        {"age": "int64", "income": "float64", "country": "object"}

    Flags three kinds of problems: a column you expected is missing, a
    column exists that you didn't account for, or a column's dtype
    doesn't match what you specified (e.g. a numeric column got loaded
    as object because of a stray non-numeric value).
    """
    records = []
    actual_cols = set(df.columns)
    expected_cols = set(expected_schema.keys())

    for col in sorted(expected_cols | actual_cols):
        if col not in actual_cols:
            records.append(
                {
                    "feature": col,
                    "status": "missing_column",
                    "expected_dtype": str(expected_schema.get(col)),
                    "actual_dtype": None,
                }
            )
            continue

        if col not in expected_cols:
            records.append(
                {
                    "feature": col,
                    "status": "unexpected_column",
                    "expected_dtype": None,
                    "actual_dtype": str(df[col].dtype),
                }
            )
            continue

        expected = expected_schema[col]
        actual_dtype = df[col].dtype
        try:
            matches = actual_dtype == np.dtype(expected)
        except TypeError:
            matches = str(actual_dtype) == str(expected)

        records.append(
            {
                "feature": col,
                "status": "ok" if matches else "dtype_mismatch",
                "expected_dtype": str(expected),
                "actual_dtype": str(actual_dtype),
            }
        )

    return pd.DataFrame(records)


def range_validation(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Validates numeric ranges and allowed categorical values.

    rules: dict mapping column_name -> constraint dict. Supported keys
    per column are "min", "max", and "allowed":

        {
            "age": {"min": 0, "max": 120},
            "income": {"min": 0},
            "status": {"allowed": ["active", "closed", "pending"]},
        }

    Each constraint you provide produces one row in the result, so a
    column with both "min" and "max" gets checked as two separate rules.
    """
    records = []
    for col, constraint in rules.items():
        if col not in df.columns:
            records.append(
                {
                    "feature": col,
                    "check": "column_missing",
                    "violation_count": None,
                    "violation_pct": None,
                }
            )
            continue

        series = df[col]

        if "min" in constraint:
            mask = series < constraint["min"]
            records.append(
                {
                    "feature": col,
                    "check": f"min >= {constraint['min']}",
                    "violation_count": int(mask.sum()),
                    "violation_pct": round(mask.mean() * 100, 2),
                }
            )

        if "max" in constraint:
            mask = series > constraint["max"]
            records.append(
                {
                    "feature": col,
                    "check": f"max <= {constraint['max']}",
                    "violation_count": int(mask.sum()),
                    "violation_pct": round(mask.mean() * 100, 2),
                }
            )

        if "allowed" in constraint:
            mask = ~series.isin(constraint["allowed"]) & series.notna()
            records.append(
                {
                    "feature": col,
                    "check": "allowed_values",
                    "violation_count": int(mask.sum()),
                    "violation_pct": round(mask.mean() * 100, 2),
                }
            )

    return pd.DataFrame(records)


_CROSS_FIELD_OPERATORS = {
    ">=": op.ge,
    ">": op.gt,
    "<=": op.le,
    "<": op.lt,
    "==": op.eq,
    "!=": op.ne,
}


def cross_field_validation(df: pd.DataFrame, rules: list[tuple[str, str, str]]) -> pd.DataFrame:
    """
    Validates logical relationships between two columns in the same row.

    rules: list of (col_a, operator, col_b) tuples, e.g.
        [("end_date", ">=", "start_date"), ("net_income", "<=", "gross_income")]

    Supported operators: ">=", ">", "<=", "<", "==", "!="

    This is the hardest of the three validation functions to make fully
    generic, since cross-field logic is inherently dataset-specific. What's
    generic here is the mechanism: give it any two columns and a comparison
    operator, and it will count how many rows violate that relationship.
    Rows where either column is missing are skipped rather than flagged.
    """
    records = []
    for col_a, operator_str, col_b in rules:
        rule_label = f"{col_a} {operator_str} {col_b}"

        if col_a not in df.columns or col_b not in df.columns:
            records.append(
                {
                    "rule": rule_label,
                    "violation_count": None,
                    "violation_pct": None,
                    "note": "one or both columns missing",
                }
            )
            continue

        if operator_str not in _CROSS_FIELD_OPERATORS:
            records.append(
                {
                    "rule": rule_label,
                    "violation_count": None,
                    "violation_pct": None,
                    "note": "unsupported operator",
                }
            )
            continue

        valid_mask = df[col_a].notna() & df[col_b].notna()
        satisfied = _CROSS_FIELD_OPERATORS[operator_str](df[col_a], df[col_b])
        violation_mask = valid_mask & ~satisfied

        records.append(
            {
                "rule": rule_label,
                "violation_count": int(violation_mask.sum()),
                "violation_pct": round(violation_mask.mean() * 100, 2),
                "note": None,
            }
        )

    return pd.DataFrame(records)


# ==========================================
# 5. MASTER WRAPPER
# ==========================================


def run_data_audit(
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
    expected_schema: dict | None = None,
    range_rules: dict | None = None,
    cross_field_rules: list[tuple[str, str, str]] | None = None,
) -> dict:
    """
    Master wrapper that executes all audit checks and returns a dictionary
    of reports. expected_schema, range_rules, and cross_field_rules are
    optional: pass them in once you know what "valid" looks like for this
    specific dataset, and their reports get added automatically.
    """
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    results = {
        "schema": schema_report(df),
        "target": target_report(df, target_col),
        "id": id_report(df, id_col),
        "duplicates": duplicate_report(df, id_col),
        "missingness": missingness_report(df),
        "hidden_missingness": hidden_missing_report(df),
        "category_normalization": category_normalization_check(df, categorical_cols),
        "infinite_values": infinite_value_report(df),
        "cardinality": cardinality_report(df),
        "constant_features": constant_feature_report(df),
        "numeric_profile": numeric_profile(df),
    }

    if expected_schema is not None:
        results["schema_validation"] = schema_validation(df, expected_schema)
    if range_rules is not None:
        results["range_validation"] = range_validation(df, range_rules)
    if cross_field_rules is not None:
        results["cross_field_validation"] = cross_field_validation(df, cross_field_rules)

    return results