"""Exercise persisted membership across notebooks and reject unsafe mappings."""

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pandas.testing import assert_frame_equal

from ml_gearbox import create_split, load_split


@pytest.fixture
def frame():
    return pd.DataFrame(
        {"target": [0] * 80 + [1] * 20, "value": np.arange(100)},
        index=[f"original-{i}" for i in range(100)],
    )


def test_no_id_roundtrip_restores_membership_after_feature_changes(frame, tmp_path):
    original = frame.copy(deep=True)
    path = tmp_path / "split.parquet"
    result = create_split(frame, stratify_col="target", path=path)
    assert_frame_equal(frame, original)
    assert result["id_col"] == "_row_id"
    assert result["df"]["_row_id"].is_unique
    assert result["df"]["_row_id"].notna().all()
    assert set(result["assignments"].columns) == {"row_id", "split"}

    # A fresh notebook reads the saved source, then reorders and engineers features.
    prepared_path = tmp_path / "prepared.parquet"
    result["df"].to_parquet(prepared_path, index=False)
    reloaded = pd.read_parquet(prepared_path).sample(frac=1, random_state=9)
    reloaded["squared"] = reloaded["value"] ** 2
    reloaded = reloaded.drop(columns="target")
    before = reloaded.copy(deep=True)
    parts = load_split(reloaded, path)
    assert_frame_equal(reloaded, before)
    for label, positions in [("train", result["trainidx"]), ("test", result["testidx"])]:
        keys = result["df"].iloc[positions]["_row_id"]
        assert_frame_equal(parts[label], reloaded.loc[reloaded["_row_id"].isin(keys)])
    assert len(parts["train"]) == 80
    assert len(parts["test"]) == 20


def test_indices_cover_every_position_once_and_stratification_balances_classes(frame, tmp_path):
    result = create_split(frame, stratify_col="target", path=tmp_path / "split.parquet")
    train, test = result["trainidx"], result["testidx"]
    assert not np.intersect1d(train, test).size
    np.testing.assert_array_equal(np.sort(np.concatenate([train, test])), np.arange(len(frame)))
    for indices in [train, test]:
        assert result["df"].iloc[indices]["target"].mean() == pytest.approx(0.2)


def test_existing_id_and_fixed_seed_are_reproducible(frame, tmp_path):
    frame["transaction_id"] = np.arange(100, 200)
    a = create_split(frame, id_col="transaction_id", random_state=7, path=tmp_path / "a.parquet")
    b = create_split(frame, id_col="transaction_id", random_state=7, path=tmp_path / "b.parquet")
    assert "_row_id" not in a["df"]
    assert_frame_equal(a["df"], frame)
    assert_frame_equal(a["assignments"], b["assignments"])
    np.testing.assert_array_equal(a["trainidx"], b["trainidx"])
    np.testing.assert_array_equal(a["testidx"], b["testidx"])
    parts = load_split(frame, tmp_path / "a.parquet")
    assert len(parts["train"]) + len(parts["test"]) == len(frame)


def test_generated_keys_are_reused(frame, tmp_path):
    a = create_split(frame, path=tmp_path / "a.parquet")
    b = create_split(a["df"], path=tmp_path / "b.parquet")
    assert_frame_equal(a["df"], b["df"])
    assert_frame_equal(a["assignments"], b["assignments"])


def test_group_split_keeps_related_rows_together(frame, tmp_path):
    frame["customer_id"] = np.repeat(np.arange(10), 10)
    result = create_split(frame, group_col="customer_id", path=tmp_path / "split.parquet")
    train_groups = set(result["df"].iloc[result["trainidx"]]["customer_id"])
    test_groups = set(result["df"].iloc[result["testidx"]]["customer_id"])
    assert train_groups.isdisjoint(test_groups)
    assert len(test_groups) == 2
    assert train_groups | test_groups == set(frame["customer_id"])


@pytest.mark.parametrize("bad_id", [[1] * 100, list(range(99)) + [None]])
def test_create_rejects_ambiguous_ids(frame, tmp_path, bad_id):
    frame["id"] = bad_id
    with pytest.raises(ValueError):
        create_split(frame, id_col="id", path=tmp_path / "split.parquet")
    assert not (tmp_path / "split.parquet").exists()


@pytest.mark.parametrize("column", ["id_col", "stratify_col", "group_col"])
def test_create_rejects_missing_supplied_columns(frame, tmp_path, column):
    with pytest.raises((ValueError, KeyError), match="missing"):
        create_split(frame, **{column: "missing"}, path=tmp_path / "split.parquet")


@pytest.mark.parametrize("size", [0, 1.0, -1, 100, True, "0.2"])
def test_create_rejects_invalid_sizes(frame, tmp_path, size):
    with pytest.raises((ValueError, TypeError)):
        create_split(frame, test_size=size, path=tmp_path / "split.parquet")


def test_cannot_combine_stratification_and_grouping(frame, tmp_path):
    with pytest.raises(ValueError):
        create_split(frame, stratify_col="target", group_col="target", path=tmp_path / "split.parquet")


def test_small_class_raises_instead_of_silently_using_random_split(frame, tmp_path):
    frame["target"] = [0] * 99 + [1]
    with pytest.raises(ValueError):
        create_split(frame, stratify_col="target", path=tmp_path / "split.parquet")


def test_single_group_and_missing_strategy_values_are_rejected(frame, tmp_path):
    frame["group"] = "only"
    with pytest.raises(ValueError):
        create_split(frame, group_col="group", path=tmp_path / "group.parquet")
    frame.loc[frame.index[0], "target"] = np.nan
    with pytest.raises(ValueError):
        create_split(frame, stratify_col="target", path=tmp_path / "stratified.parquet")


def test_existing_file_requires_explicit_overwrite(frame, tmp_path):
    path = tmp_path / "split.parquet"
    create_split(frame, path=path)
    original_bytes = path.read_bytes()
    with pytest.raises(FileExistsError):
        create_split(frame, path=path)
    assert path.read_bytes() == original_bytes
    result = create_split(frame, path=path, overwrite=True)
    assert len(load_split(result["df"], path)["train"]) == 80


def test_strict_loader_rejects_filtered_rows_but_subset_mode_preserves_membership(frame, tmp_path):
    path = tmp_path / "split.parquet"
    result = create_split(frame, path=path)
    train_only = result["df"].iloc[result["trainidx"]].copy()
    with pytest.raises(ValueError):
        load_split(train_only, path)
    parts = load_split(train_only, path, allow_subset=True)
    assert_frame_equal(parts["train"], train_only)
    assert parts["test"].empty


@pytest.mark.parametrize("problem", ["unknown", "duplicate", "null", "missing_column"])
def test_loader_rejects_invalid_or_absent_keys(frame, tmp_path, problem):
    path = tmp_path / "split.parquet"
    result = create_split(frame, path=path)
    changed = result["df"].copy()
    if problem == "missing_column":
        changed = changed.drop(columns="_row_id")
    else:
        changed.loc[changed.index[0], "_row_id"] = {
            "unknown": "another-dataset",
            "duplicate": changed["_row_id"].iloc[1],
            "null": None,
        }[problem]
    with pytest.raises((ValueError, KeyError)):
        load_split(changed, path, allow_subset=True)


def test_regenerated_keys_cannot_match_previous_split(frame, tmp_path):
    path = tmp_path / "original.parquet"
    create_split(frame, path=path)
    regenerated = create_split(frame, path=tmp_path / "regenerated.parquet")
    with pytest.raises(ValueError):
        load_split(regenerated["df"], path)


def test_integer_ids_allow_nullable_roundtrip_but_forbid_string_coercion(frame, tmp_path):
    frame["id"] = pd.array(np.arange(100), dtype="Int64")
    path = tmp_path / "integer.parquet"
    create_split(frame, id_col="id", path=path)
    compatible = frame.copy()
    compatible["id"] = compatible["id"].astype("int64")
    assert len(load_split(compatible, path)["train"]) == 80
    incompatible = frame.copy()
    incompatible["id"] = incompatible["id"].astype(str)
    with pytest.raises(ValueError):
        load_split(incompatible, path)


def test_large_unsigned_ids_preserve_exact_identity(frame, tmp_path):
    frame["id"] = np.arange(100, dtype="uint64") + np.uint64(2**63 + 2**53)
    path = tmp_path / "large_ids.parquet"
    result = create_split(frame, id_col="id", path=path)
    frame["id"] = frame["id"].astype("UInt64")
    parts = load_split(frame, path)
    expected = set(result["df"].iloc[result["trainidx"]]["id"])
    assert set(parts["train"]["id"]) == expected


def test_date_ids_roundtrip_through_arrow_backend(frame, tmp_path):
    frame["id"] = [date(2020, 1, 1) + timedelta(days=i) for i in range(len(frame))]
    path = tmp_path / "dates.parquet"
    result = create_split(frame, id_col="id", path=path)
    prepared_path = tmp_path / "prepared.parquet"
    result["df"].to_parquet(prepared_path, index=False)
    reloaded = pd.read_parquet(prepared_path, dtype_backend="pyarrow")
    parts = load_split(reloaded, path)
    expected = set(result["df"].iloc[result["trainidx"]]["id"])
    assert set(parts["train"]["id"]) == expected


def test_stratification_ignores_unused_category_levels(frame, tmp_path):
    frame["target"] = pd.Categorical(frame["target"], categories=[0, 1, 2])
    result = create_split(frame, stratify_col="target", path=tmp_path / "split.parquet")
    assert len(result["testidx"]) == 20


def test_string_ids_keep_leading_zeros_and_duplicate_dataframe_index_is_allowed(frame, tmp_path):
    frame["id"] = [f"{i:03d}" for i in range(100)]
    frame.index = [0] * len(frame)
    path = tmp_path / "split.parquet"
    result = create_split(frame, id_col="id", path=path)
    parts = load_split(frame, path)
    expected = set(result["df"].iloc[result["trainidx"]]["id"])
    assert set(parts["train"]["id"]) == expected
    changed = frame.copy()
    changed["id"] = changed["id"].astype(int)
    with pytest.raises(ValueError):
        load_split(changed, path)


def _rewrite_mapping(path, mutate_data=None, mutate_metadata=None, remove_metadata=False):
    table = pq.read_table(path)
    metadata = dict(table.schema.metadata or {})
    if mutate_data:
        mapping = mutate_data(table.to_pandas())
        table = pa.Table.from_pandas(mapping, preserve_index=False)
    if mutate_metadata:
        details = json.loads(metadata[b"ml_gearbox.split"])
        mutate_metadata(details)
        metadata[b"ml_gearbox.split"] = json.dumps(details).encode()
    if remove_metadata:
        metadata.pop(b"ml_gearbox.split", None)
    pq.write_table(table.replace_schema_metadata(metadata), path)


@pytest.mark.parametrize("problem", ["duplicate_key", "null_key", "bad_label", "null_label", "only_train", "missing_metadata", "row_count", "schema_version", "malformed_key_family"])
def test_loader_rejects_corrupted_saved_mapping(frame, tmp_path, problem):
    path = tmp_path / "split.parquet"
    result = create_split(frame, path=path)
    if problem == "duplicate_key":
        _rewrite_mapping(path, lambda df: df.assign(row_id=[df.row_id.iloc[0]] * len(df)))
    elif problem == "null_key":
        _rewrite_mapping(path, lambda df: df.assign(row_id=None))
    elif problem == "bad_label":
        _rewrite_mapping(path, lambda df: df.assign(split="validation"))
    elif problem == "null_label":
        _rewrite_mapping(path, lambda df: df.assign(split=None))
    elif problem == "only_train":
        _rewrite_mapping(path, lambda df: df.assign(split="train"))
    elif problem == "missing_metadata":
        _rewrite_mapping(path, remove_metadata=True)
    elif problem == "row_count":
        _rewrite_mapping(path, mutate_metadata=lambda meta: meta.update(row_count=999))
    elif problem == "malformed_key_family":
        _rewrite_mapping(path, mutate_metadata=lambda meta: meta.update(key_family=[]))
    else:
        _rewrite_mapping(path, mutate_metadata=lambda meta: meta.update(schema_version=999))
    with pytest.raises(ValueError):
        load_split(result["df"], path)
