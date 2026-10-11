"""Create and restore persistent, row-keyed train/test splits.

Install ``ml-gearbox[ml,parquet]`` to create splits, or the ``parquet`` extra
alone to restore them. Optional dependencies are imported only when needed.
"""

from __future__ import annotations

import json
import numbers
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from pandas.api import types as pd_types

_METADATA_KEY = b"ml_gearbox.split"
_SCHEMA_VERSION = 1
_KEY_FAMILIES = {
    "string",
    "integer",
    "floating",
    "boolean",
    "datetime",
    "timedelta",
    "date",
    "binary",
    "decimal",
}


def _parquet_dependencies():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Persistent splits require pyarrow; install 'ml-gearbox[parquet]'."
        ) from exc
    return pa, pq


def _validate_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not df.columns.is_unique:
        raise ValueError("Dataframe column names must be unique.")


def _validate_column(df: pd.DataFrame, column: str, argument: str) -> None:
    if not isinstance(column, str) or not column:
        raise ValueError(f"{argument} must be a nonempty column name.")
    if column not in df.columns:
        raise ValueError(f"{argument} column {column!r} does not exist.")


def _key_family(keys: pd.Series) -> str:
    """Identify semantic key types without coercing strings or numeric IDs."""
    dtype = keys.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        values = keys.astype(object)
        if values.empty:
            values = pd.Series(dtype.categories)
        return _key_family(values)
    # Pandas considers Arrow dates datetime-like, but their scalar values
    # remain datetime.date, just like object-date keys read without Arrow.
    if isinstance(dtype, pd.ArrowDtype):
        import pyarrow as pa

        if pa.types.is_date(dtype.pyarrow_dtype):
            return "date"
    if pd_types.is_bool_dtype(dtype):
        return "boolean"
    if pd_types.is_integer_dtype(dtype):
        return "integer"
    if pd_types.is_float_dtype(dtype):
        return "floating"
    if pd_types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd_types.is_timedelta64_dtype(dtype):
        return "timedelta"
    if isinstance(dtype, pd.StringDtype):
        return "string"

    inferred = pd_types.infer_dtype(keys, skipna=True)
    family = {
        "string": "string",
        "unicode": "string",
        "bytes": "binary",
        "integer": "integer",
        "floating": "floating",
        "boolean": "boolean",
        "datetime": "datetime",
        "datetime64": "datetime",
        "timedelta": "timedelta",
        "timedelta64": "timedelta",
        "date": "date",
        "decimal": "decimal",
        "empty": "empty",
    }.get(inferred)
    if family is None:
        raise ValueError(
            "Row keys must have one supported scalar type; mixed types, "
            "complex numbers, and containers are not supported."
        )
    return family


def _validate_keys(keys: pd.Series, description: str) -> str:
    if keys.isna().any():
        raise ValueError(f"{description} contains missing row keys.")
    family = _key_family(keys)
    try:
        duplicated = keys.duplicated().any()
    except TypeError as exc:
        raise ValueError(
            f"{description} must contain hashable scalar row keys."
        ) from exc
    if duplicated:
        raise ValueError(f"{description} contains duplicate row keys.")
    return family


def _validate_test_size(test_size):
    if isinstance(test_size, bool):
        raise ValueError(
            "test_size must be a fraction between 0 and 1 or a positive count."
        )
    if isinstance(test_size, numbers.Integral):
        if test_size <= 0:
            raise ValueError("An integer test_size must be positive.")
        return int(test_size)
    if isinstance(test_size, numbers.Real) and 0 < test_size < 1:
        return float(test_size)
    raise ValueError(
        "test_size must be a fraction between 0 and 1 or a positive count."
    )


def _validate_seed(random_state):
    if random_state is None:
        return None
    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, numbers.Integral)
        or not 0 <= random_state < 2**32
    ):
        raise ValueError("random_state must be None or an integer from 0 to 2**32 - 1.")
    return int(random_state)


def create_split(
    df: pd.DataFrame,
    *,
    id_col: str | None = None,
    stratify_col: str | None = None,
    group_col: str | None = None,
    test_size: float | int = 0.2,
    random_state: int | None = 42,
    path: str | Path = "split.parquet",
    overwrite: bool = False,
) -> dict:
    """Create train/test positions and persist each row's split membership.

    ``id_col`` identifies observations and must be unique and non-null. If it
    is omitted, an existing ``_row_id`` is reused; otherwise UUID strings are
    added as ``_row_id`` on a copy of ``df``. Save the returned dataframe and
    retain that key column in later notebooks. The input is never modified.

    ``stratify_col`` balances category proportions. ``group_col`` keeps every
    group in one partition; its test_size counts groups, so the fraction of
    test *rows* can differ. These strategies cannot be combined. Without
    either column the split is random. Strategy values must be non-null.

    ``test_size`` is a fraction in (0, 1) or a positive integer count.
    ``random_state`` fixes membership for an unchanged, ordered dataframe.
    Existing files are refused unless ``overwrite=True``.

    Returns a dictionary with ``df`` (the prepared copy), ``id_col`` (the
    effective key column), ``trainidx`` and ``testidx`` (sorted zero-based
    positions for ``df.iloc``), and ``assignments`` (``row_id`` / ``split``).
    Positions apply only to this snapshot; the Parquet mapping uses row keys
    and explicit metadata so it survives sorting and added feature columns.
    """
    _validate_dataframe(df)
    if len(df) < 2:
        raise ValueError("A split requires at least two rows.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")
    test_size = _validate_test_size(test_size)
    random_state = _validate_seed(random_state)
    if stratify_col is not None and group_col is not None:
        raise ValueError("stratify_col and group_col cannot be used together.")
    if id_col is not None:
        _validate_column(df, id_col, "id_col")
    for argument, column in (("stratify_col", stratify_col), ("group_col", group_col)):
        if column is not None:
            _validate_column(df, column, argument)
            if df[column].isna().any():
                raise ValueError(
                    f"{argument} column {column!r} contains missing values."
                )

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Split file already exists: {destination}. Use overwrite=True to replace it."
        )
    prepared = df.copy()
    effective_id = id_col if id_col is not None else "_row_id"
    if effective_id not in prepared.columns:
        prepared[effective_id] = [str(uuid4()) for _ in range(len(prepared))]
    key_family = _validate_keys(prepared[effective_id], f"ID column {effective_id!r}")

    if group_col is not None:
        strategy, strategy_col = "group", group_col
        if prepared[group_col].nunique() < 2:
            raise ValueError("A group split requires at least two distinct groups.")
    elif stratify_col is not None:
        strategy, strategy_col = "stratified", stratify_col
        counts = prepared[stratify_col].value_counts()
        if (counts[counts > 0] < 2).any():
            raise ValueError("Every stratification class must have at least two rows.")
    else:
        strategy, strategy_col = "random", None

    pa, pq = _parquet_dependencies()
    try:
        from sklearn.model_selection import GroupShuffleSplit, train_test_split
    except ImportError as exc:
        raise ImportError(
            "Creating splits requires scikit-learn; install 'ml-gearbox[ml,parquet]'."
        ) from exc

    positions = np.arange(len(prepared))
    try:
        if group_col is not None:
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
            trainidx, testidx = next(
                splitter.split(positions, groups=prepared[group_col])
            )
        else:
            trainidx, testidx = train_test_split(
                positions,
                test_size=test_size,
                random_state=random_state,
                stratify=prepared[stratify_col] if stratify_col is not None else None,
            )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unable to create {strategy} split: {exc}") from exc

    trainidx, testidx = np.sort(trainidx), np.sort(testidx)
    combined = np.concatenate((trainidx, testidx))
    if (
        not len(trainidx)
        or not len(testidx)
        or len(combined) != len(prepared)
        or not np.array_equal(np.sort(combined), positions)
    ):
        raise ValueError(
            "The split must have two nonempty, disjoint partitions covering every row."
        )

    labels = np.full(len(prepared), "train", dtype="<U5")
    labels[testidx] = "test"
    assignments = pd.DataFrame(
        {
            "row_id": prepared[effective_id].reset_index(drop=True).copy(),
            "split": labels,
        }
    )
    metadata = {
        "schema_version": _SCHEMA_VERSION,
        "id_col": effective_id,
        "key_dtype": str(prepared[effective_id].dtype),
        "key_family": key_family,
        "strategy": strategy,
        "strategy_col": strategy_col,
        "test_size": test_size,
        "random_state": random_state,
        "row_count": len(prepared),
    }
    table = pa.Table.from_pandas(assignments, preserve_index=False)
    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata[_METADATA_KEY] = json.dumps(metadata).encode("utf-8")
    table = table.replace_schema_metadata(schema_metadata)
    # Exclusive creation also closes the race between checking and writing.
    with destination.open("wb" if overwrite else "xb") as output:
        pq.write_table(table, output)
    return {
        "df": prepared,
        "id_col": effective_id,
        "trainidx": trainidx,
        "testidx": testidx,
        "assignments": assignments,
    }


def _read_metadata(table) -> dict:
    raw_metadata = (table.schema.metadata or {}).get(_METADATA_KEY)
    if raw_metadata is None:
        raise ValueError("Split Parquet file is missing ml_gearbox split metadata.")
    try:
        metadata = json.loads(raw_metadata.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Split metadata is not valid UTF-8 JSON.") from exc
    required = {
        "schema_version",
        "id_col",
        "key_dtype",
        "key_family",
        "strategy",
        "strategy_col",
        "test_size",
        "random_state",
        "row_count",
    }
    if not isinstance(metadata, dict) or not required.issubset(metadata):
        raise ValueError("Split metadata is missing required fields.")
    version = metadata["schema_version"]
    if type(version) is not int or version != _SCHEMA_VERSION:
        raise ValueError(f"Unsupported split schema version: {version!r}.")
    for field in ("id_col", "key_dtype"):
        if not isinstance(metadata[field], str) or not metadata[field]:
            raise ValueError(f"Split metadata has an invalid {field}.")
    if (
        not isinstance(metadata["key_family"], str)
        or metadata["key_family"] not in _KEY_FAMILIES
    ):
        raise ValueError("Split metadata has an unsupported key_family.")
    if type(metadata["row_count"]) is not int or metadata["row_count"] < 2:
        raise ValueError("Split metadata has an invalid row_count.")
    strategy = metadata["strategy"]
    if strategy not in ("random", "stratified", "group"):
        raise ValueError("Split metadata has an unsupported strategy.")
    column = metadata["strategy_col"]
    if (strategy == "random" and column is not None) or (
        strategy != "random" and (not isinstance(column, str) or not column)
    ):
        raise ValueError("Split metadata has an invalid strategy_col.")
    size = _validate_test_size(metadata["test_size"])
    if isinstance(size, int) and size >= metadata["row_count"]:
        raise ValueError("Split metadata test_size must leave training rows.")
    _validate_seed(metadata["random_state"])
    return metadata


def load_split(
    df: pd.DataFrame,
    path: str | Path = "split.parquet",
    *,
    allow_subset: bool = False,
) -> dict:
    """Restore train and test dataframes using a saved row-key mapping.

    The key column is read from explicit Parquet metadata. Row keys must be
    unique, non-null, compatible with the saved key type, and known to the
    mapping. Keys are never regenerated or converted to strings. Ordinary
    dtype changes such as int64 to nullable Int64 are accepted; string IDs
    such as ``"001"`` cannot be replaced with numeric IDs.

    By default, ``df`` must contain exactly the saved set of observations.
    ``allow_subset=True`` permits deliberate filtering, including an empty
    dataframe, while still rejecting unknown keys. Either returned partition
    may then be empty. The complete saved mapping must always contain both
    train and test rows.

    Returns ``{"train": train_df, "test": test_df}``, preserving the input's
    order, index, and columns in independent copies. Sorting, resetting the
    index, and adding features are safe when the original keys are retained.
    Key validation checks membership, not unchanged feature or target values.
    """
    _validate_dataframe(df)
    if not isinstance(allow_subset, bool):
        raise TypeError("allow_subset must be a boolean.")
    _, pq = _parquet_dependencies()
    table = pq.ParquetFile(path).read()
    metadata = _read_metadata(table)
    if table.column_names != ["row_id", "split"]:
        raise ValueError(
            "Split mapping must contain exactly the columns row_id and split."
        )
    assignments = table.to_pandas()
    if len(assignments) != metadata["row_count"]:
        raise ValueError("Split metadata row_count does not match the mapping length.")
    saved_family = _validate_keys(assignments["row_id"], "Saved mapping")
    if saved_family != metadata["key_family"]:
        raise ValueError("Saved row-key type does not match the split metadata.")
    if assignments["split"].isna().any():
        raise ValueError("Saved mapping contains missing split labels.")
    if not assignments["split"].isin(["train", "test"]).all():
        raise ValueError(
            "Saved mapping contains unsupported split labels; use train and test."
        )
    if set(assignments["split"]) != {"train", "test"}:
        raise ValueError("Saved mapping must contain both train and test assignments.")

    effective_id = metadata["id_col"]
    _validate_column(df, effective_id, "saved ID")
    input_family = _validate_keys(df[effective_id], f"ID column {effective_id!r}")
    if input_family != saved_family and not (df.empty and input_family == "empty"):
        raise ValueError(
            f"Incompatible row-key types: input is {input_family}, saved keys are {saved_family}."
        )

    # Python scalar keys retain integer precision across nullable/unsigned
    # dtype changes; no merge or string coercion can silently lose identity.
    membership = dict(
        zip(assignments["row_id"].tolist(), assignments["split"].tolist())
    )
    labels = []
    for key in df[effective_id].tolist():
        if key not in membership:
            raise ValueError(
                "Input contains unknown row keys without a saved assignment."
            )
        labels.append(membership[key])
    if not allow_subset and len(df) != len(assignments):
        raise ValueError(
            "Input row keys do not cover the complete saved mapping; "
            "use allow_subset=True for deliberate filtering."
        )
    labels = np.asarray(labels, dtype="<U5")
    return {
        "train": df.loc[labels == "train"].copy(),
        "test": df.loc[labels == "test"].copy(),
    }
