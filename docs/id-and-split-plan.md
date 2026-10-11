# Optional IDs, reusable splits, and feature-engineering helpers

Status: sections 1–4 implemented for version 0.1.1. Feature-engineering helpers
in section 5 remain a deferred roadmap and are excluded from this release.

Before this change, `duplicate_report` already accepted `id_col=None`, but
`run_data_audit` required an ID and `splitting.py` was empty. This design extends
those parts while keeping feature engineering as a separate roadmap. See the
README for the implemented API and the notebook example explained line by line.

## 1. Audit dataframes without an ID

Change the wrapper to `run_data_audit(df, target_col, id_col=None, ...)`.

- With an ID supplied, retain the existing identifier and duplicate reports.
- With no ID supplied, return `results["id"] = None` and run full-row duplicate
  detection. Keep the other audit reports unchanged.
- A supplied column that does not exist is an error with a clear message.
  Do not silently treat a misspelled column as an omitted ID.
- Do not infer an ID from a column name or the dataframe index, and do not add
  synthetic IDs inside the audit. Adding a unique column before checking full-row
  duplicates would hide duplicate observations.

Keep `id_report(df, id_col)` as the explicit identifier-only check. Update the
README to explain that the wrapper's ID report can be unavailable.

## 2. Persist row identity separately from split strategy

For saved splits, every observation needs a unique, non-null key:

- Use `id_col` when provided, validating uniqueness and missing values first.
- Otherwise reuse a previously generated `_row_id` column, validating it too.
- If neither exists, generate `_row_id` once on a copy of the dataframe. Use
  UUID strings so regenerating keys for another dataset cannot accidentally
  match old split assignments.
- Return the dataframe containing the generated keys. Save it as the canonical
  source dataset before another notebook uses the split. Keep `_row_id` through
  sorting, filtering, joins, and feature engineering; exclude it from model inputs.

Keeping keys through a join assumes each observation remains unique. A
one-to-many join duplicates keys and needs an explicit identity policy before
loading or creating a row-level split.

Never reconstruct keys from row positions or a newly reset index in another
notebook. Without preserved IDs or an unchanged source snapshot, there is no
general way to recover the identity of an arbitrary row, particularly when
identical records exist. Row-content hashing also changes when features change.

If `customer_id` repeats because there are multiple observations per customer,
it is a grouping column rather than a unique row ID. Use a transaction ID or
generated `_row_id` for row identity and `group_col="customer_id"` for splitting.

## 3. Create indices and save membership

API in `splitting.py`:

```python
create_split(
    df,
    *,
    id_col=None,
    stratify_col=None,
    group_col=None,
    test_size=0.2,
    random_state=42,
    path="split.parquet",
    overwrite=False,
) -> dict
```

The returned dictionary contains:

| Key | Meaning |
| --- | --- |
| `df` | Copy of the input, including a generated row key if needed |
| `id_col` | Effective key column, for example `transaction_id` or `_row_id` |
| `trainidx` | Zero-based row positions for `df.iloc[trainidx]` in this snapshot |
| `testidx` | Zero-based row positions for `df.iloc[testidx]` in this snapshot |
| `assignments` | One row per observation with `row_id` and `split` |

Save `assignments` to `path` with labels `train` and `test`. Positional indices
are conveniences for the current dataframe; saved membership uses row keys.
Refuse an existing destination unless `overwrite=True` is explicitly requested.

| Argument | Purpose |
| --- | --- |
| `id_col` | Identifies each observation when restoring the split |
| `stratify_col` | Preserves category proportions, commonly the classification target |
| `group_col` | Keeps all rows of one entity together in a single split |

With neither strategy column, use a random split. For stratification, use
[`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).
For groups, use
[`GroupShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html):
its test fraction applies to groups, so the row fraction can differ.
Initially reject requests combining stratification and grouping rather than
silently ignoring one. Chronological splitting and importing an existing column
of train/test labels can be later extensions with separate explicit arguments.

Validate missing strategy columns, missing strategy values, invalid split sizes,
too few groups, and class counts too small for the requested stratification.
Propagate a useful error instead of falling back to an unstratified split.
Require complete coverage, disjoint assignments, and nonempty train and test
partitions.

Store schema version, effective ID column, key dtype, strategy, strategy column,
requested test size, random seed, and original row count in Parquet metadata.
Do not rely on `DataFrame.attrs` as an implicit storage contract. The mapping
stores membership, while metadata explains how it was created.

## 4. Restore train and test dataframes in other notebooks

```python
load_split(df, path="split.parquet", *, allow_subset=False) -> dict
```

Read the mapping and its metadata, obtain the effective key column, validate it,
and map each input key to its saved label. Return `{"train": train_df,
"test": test_df}`. Callers can select either result without running a new split.

- Preserve the input dataframe's order, index, and feature columns.
- Accept reordered rows and changed feature columns when the row keys remain.
- Reject missing/null/duplicate input keys, incompatible key types, duplicate
  mapping keys, missing labels, unsupported labels, and unknown input keys.
- Permit equivalent key representations after ordinary Parquet dtype changes,
  such as nullable versus non-nullable integers. Reject lossy coercion such as
  converting a string ID `"001"` to numeric `1`.
- Validate metadata row count against mapping length and require both train and
  test labels in the complete saved mapping.
- By default require the same key set as the saved mapping. With
  `allow_subset=True`, allow deliberate filtering of the original rows; every
  remaining row still needs a saved assignment. Empty partitions are possible
  for such subsets and should be documented.
- Never regenerate IDs in this loader or silently drop unmatched observations.

Identity validation proves membership can be restored; it does not establish
that feature values or targets have stayed unchanged. Version the canonical
source dataset when source-record changes matter.

Notebook workflow when the raw data has no ID:

```python
# Notebook 1: audit the original data, then create and preserve row keys.
audit = run_data_audit(raw_df, target_col="target")
result = create_split(raw_df, stratify_col="target", path="split.parquet")
result["df"].to_parquet("prepared.parquet", index=False)
train_df = result["df"].iloc[result["trainidx"]].copy()
test_df = result["df"].iloc[result["testidx"]].copy()

# Notebook 2: retain the saved row keys when engineering features.
df = pd.read_parquet("prepared.parquet")
parts = load_split(df, "split.parquet")
train_df, test_df = parts["train"], parts["test"]
```

Use the existing optional `ml` extra for scikit-learn and add a `parquet` extra
with `pyarrow`. Document `pip install -e ".[ml,parquet]"`. Pandas requires
[a Parquet backend](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html).
Keep optional imports local so audit-only users can use the package without
installing the split dependencies.

## 5. Suggested reusable feature-engineering functions

Prefer explicit column lists, return copies, retain row keys and indices, and
reject output-column collisions unless replacement is requested.

| Priority | Helpers | Useful behavior |
| --- | --- | --- |
| First | `normalize_missing_values`, `normalize_categories` | Convert configured sentinels and blank strings to missing; strip/case-normalize categories while preserving actual nulls |
| First | `add_missing_indicators` | Add consistent missingness flags for explicitly selected columns |
| First | `safe_ratio` | Create ratio features with explicit zero-denominator and missing-value behavior |
| Next | `fit_imputation`, `apply_imputation` | Learn medians/modes on training rows and reuse the same values elsewhere |
| Next | `fit_rare_categories`, `apply_rare_categories` | Learn a category vocabulary on train; define how rare, missing, and unseen categories are handled |
| Next | `fit_frequency_encoding`, `apply_frequency_encoding` | Learn counts/frequencies on train and specify an unseen-category default |
| Next | `fit_clip_bounds`, `apply_clip_bounds` | Learn IQR/quantile bounds on train and reuse them on test |
| Later | `fit_bins`, `apply_bins`, `numeric_transform` | Preserve training bin edges; offer explicit log/domain and out-of-range handling |
| Later | `datetime_features`, `elapsed_time`, `cyclical_encode` | Calendar components, fixed-reference durations, and configurable periodic features |

Learned transforms need serializable fitted state and separate fit/apply steps.
Fit on training rows only, including within each cross-validation fold. Preserve
the same output schema on train and test. Keep target encoding for a later phase
because it needs out-of-fold training behavior to avoid target leakage.

## 6. Implementation order and acceptance checks

1. Optional-ID auditing and README changes.
2. Split creation, generated-key persistence, loading, and optional dependencies.
3. Notebook examples and focused tests.
4. Stateless feature helpers first, followed by fitted transformations.

Checks for the ID and split work should cover auditing without an ID (including
duplicate records), existing-ID behavior, invalid supplied columns, deterministic
splitting for a fixed keyed dataframe and seed, complete/disjoint partitions,
class proportions, group separation, and an actual Parquet round trip. Verify
loading after shuffle/reset_index and after feature changes; reject missing,
duplicate, unknown, or regenerated keys and malformed mappings. Exercise strict
coverage and intentional subsets, and confirm the original dataframe is not
modified.
