# ml-gearbox

A reusable set of pandas-based functions for inspecting datasets and preserving train/test membership across notebooks.

- **`audit`** — is this dataset trustworthy? (missingness, duplicates, schema, validity)
- **`eda`** — what does this dataset actually look like, and how does it relate to the target?
- **`splitting`** — create a train/test split once, save its row assignments, and restore it later.

Version **0.1.1** adds optional IDs in the audit wrapper and reusable Parquet splits.

This README documents every function currently in the toolkit: what it checks, what it expects from you, what it returns, and — most importantly — **what decision to make once you see the result.** It's meant to be read like a reference manual, not front-to-back.

## Install

```bash
pip install "ml-gearbox[ml,parquet]==0.1.1"
```

The `ml` extra installs scikit-learn for splitting; `parquet` installs PyArrow for saving and reading split files. Audit and EDA work with the base package: `pip install ml-gearbox==0.1.1`.

For local development, run this from the project root, where `pyproject.toml` lives:

```bash
pip install -e ".[ml,parquet]"
```

Editable mode picks up changes to the source without reinstalling.

## Quickstart

```python
from ml_gearbox.audit import run_data_audit
from ml_gearbox.eda import iqr_outlier_summary, plot_correlation_heatmap, correlation_matrix

# Step 1: is the data trustworthy?
audit = run_data_audit(df, target_col="default")
print(audit["missingness"])
print(audit["duplicates"])
# If each row has a unique identifier, optionally pass id_col="transaction_id".

# Step 2: once it's clean, explore it
outliers = iqr_outlier_summary(df, columns=["age", "income"])
corr = correlation_matrix(df, columns=["age", "income", "tenure"])
plot_correlation_heatmap(corr)
```

## How to use the modules

Run **`audit` first, always**. It answers "can I trust this data at all?" Nothing in `eda` is meaningful if your dataset has silent duplicate IDs, 40% hidden missingness, or a target column that's actually a string when you expect numeric.

Once audit comes back clean (or you've fixed what it found), move to **`eda`** to understand *shape*: distributions, outliers, correlations, and how features relate to your target.

Use **`splitting`** to save the train/test membership that later notebooks will reuse. Keep its row key when adding features, and exclude the key from model inputs. Fit any transformations that learn statistics on training rows only.

### Suggested order of operations

1. `schema_report` — get the lay of the land (dtypes, missingness, uniqueness, in one table)
2. `id_report` when an ID exists, plus `duplicate_report` — check row identity and duplicate observations
3. `missingness_report` + `hidden_missing_report` — where are the gaps, including disguised ones?
4. `category_normalization_check` — are your categories cleaner than they look?
5. `infinite_value_report` — any `inf` values that will silently break downstream math?
6. `cardinality_report` + `constant_feature_report` — anything uninformative or unmanageably high-cardinality?
7. `schema_validation` / `range_validation` / `cross_field_validation` — once you know the rules this dataset should follow, check it actually follows them
8. Now move to `eda.py` — distributions, outliers, correlations, target rates, and their plots

Or just call `run_data_audit(...)` to get steps 1–7 in one dictionary.

---

## `audit.py` reference

### Target, ID, and duplicates

#### `target_report(df, target_col) -> DataFrame`
- **Checks:** the class balance of your target variable.
- **Expects:** `target_col` is a column name in `df`. Works for any number of classes, including a numeric binary target.
- **Returns:** one row per class, with `count` and `percentage`.
- **Act on it:** if one class is under ~5–10%, you're dealing with class imbalance — plan for it now (stratified splits, class weights, resampling, or a metric other than accuracy) rather than after your first model mysteriously predicts one class every time.

#### `id_report(df, id_col) -> dict`
- **Checks:** whether your identifier column is actually a valid identifier.
- **Expects:** `id_col` is the column meant to uniquely identify each row (a customer ID, transaction ID, etc.).
- **Returns:** `total_rows`, `unique_ids`, `missing_ids`, `duplicate_ids`.
- **Act on it:** `missing_ids > 0` or `duplicate_ids > 0` means something is wrong upstream — a broken join, a bad extract, or two systems using the same ID space differently. Investigate the source before doing anything else; don't just drop rows and move on.

#### `duplicate_report(df, id_col=None) -> dict`
- **Checks:** duplicate records at two levels. `full_row_duplicates` counts rows that are identical across *every* column (safe to drop — almost always a copy-paste or re-import artifact). If you pass `id_col`, it additionally reports `duplicate_id_rows` (rows sharing an ID) and, critically, `conflicting_id_rows` — rows that share the same ID but have **different** values elsewhere.
- **Expects:** `id_col` is optional but strongly recommended if your dataset has one; without it you only get the full-row check, which will show 0 on almost any dataset that has an ID column (since the ID makes every row technically unique).
- **Returns:** a dict, e.g. `{"full_row_duplicates": 1, "duplicate_id_rows": 2, "conflicting_id_rows": 2}`.
- **Act on it:** `full_row_duplicates` → safe to `df.drop_duplicates()`. `conflicting_id_rows` is the dangerous one — it means the same entity has two different versions of the truth on record (e.g. two different incomes for the same customer ID). Don't silently pick one; find out which row is current/correct, usually via a timestamp column if one exists.

### Missingness

#### `missingness_report(df) -> DataFrame`
- **Checks:** standard missing values (`NaN`/`None`) per column.
- **Expects:** nothing special — works on the whole dataframe.
- **Returns:** `feature`, `missing_count`, `percentage`, only for columns with at least one missing value, sorted worst-first. Empty DataFrame if nothing's missing.
- **Act on it:** under ~5% missing → usually safe to impute (median/mode) or drop rows. 5–40% → impute carefully, consider a "was_missing" flag as its own feature since the *fact* of missingness can be predictive. Above ~40–50% → seriously consider dropping the column; imputation is mostly guessing at that point.

#### `hidden_missing_report(df) -> DataFrame`
- **Checks:** missingness that doesn't show up in `.isna()` because it's disguised as an empty or whitespace-only string (`""`, `"   "`).
- **Expects:** runs on object/category columns only.
- **Returns:** `feature`, `hidden_missing_count`, `percentage`. Empty DataFrame if none found.
- **Act on it:** any row this flags should be treated as missing, not as a valid (blank) category. Convert them to real `NaN` (`df[col] = df[col].replace(r'^\s*$', pd.NA, regex=True)`) before running `missingness_report` again, or your missingness numbers upstream are undercounting.

#### `category_normalization_check(df, columns) -> DataFrame`
- **Checks:** whether categories that look different are actually the same thing with different formatting — `"USA"`, `"usa"`, `" USA "` all collapsing into one category once stripped and lowercased.
- **Expects:** a list of categorical column names to check.
- **Returns:** `feature`, `original_unique`, `normalized_unique` — only for columns where normalizing *reduces* the category count. Empty DataFrame if formatting is already clean.
- **Act on it:** if a column shows up here, your `cardinality_report` and any category-based model feature (one-hot encoding, target encoding) is currently splitting one real-world category into several. Normalize the column (`.str.strip().str.lower()`, or a proper mapping) before encoding, not after.

#### `infinite_value_report(df) -> DataFrame`
- **Checks:** `inf` / `-inf` values in numeric columns — usually the result of a division by zero somewhere upstream (e.g. a ratio feature like `income / years_employed` where `years_employed` was 0).
- **Expects:** nothing special.
- **Returns:** `feature`, `infinite_count`. Empty DataFrame if none found.
- **Act on it:** treat these as missing (`df.replace([np.inf, -np.inf], np.nan)`) or fix the calculation that produced them. **Do this before running any quantile-based function** (see the Known Quirks section below) — `inf` values will crash `numeric_target_rate_by_quantile` and `pd.qcut`-based binning in general.

### Schema, cardinality, constants

#### `schema_report(df) -> DataFrame`
- **Checks:** a one-table overview of every column: dtype, missing count/%, and unique count/%.
- **Expects:** nothing special. This is usually your first call on a new dataset.
- **Returns:** one row per column, sorted by missingness (worst first).
- **Act on it:** use this as your map. Anything with `dtype == object` that you expected to be numeric, or `unique_pct` near 100% on a column that isn't an ID, deserves a closer look before you go further.

#### `cardinality_report(df) -> DataFrame`
- **Checks:** how many unique values each categorical (object/category) column has.
- **Expects:** nothing special.
- **Returns:** `feature`, `n_unique`, sorted highest first.
- **Act on it:** columns with very high cardinality (hundreds/thousands of unique values — free-text names, raw IDs accidentally typed as category) will blow up one-hot encoding. Use target/frequency encoding, group rare categories into "other", or drop the column if it's really just an identifier.

#### `constant_feature_report(df) -> DataFrame`
- **Checks:** how dominant the single most common value is in each column (works on *any* column, not just categorical).
- **Expects:** nothing special.
- **Returns:** `feature`, `dominant_percentage`, `is_strictly_constant` (True only if one value makes up 100%), sorted highest first.
- **Act on it:** `is_strictly_constant == True` → the column carries zero information for modeling; drop it. Dominant but not strictly constant (e.g. 97%) → the column will likely have very low predictive power and near-zero variance; consider dropping it too, or check whether the rare values are actually meaningful outliers worth keeping.

#### `numeric_profile(df) -> DataFrame`
- **Checks:** an enhanced `.describe()` for numeric columns — adds median and skew on top of the standard mean/std/min/max/quartiles.
- **Expects:** nothing special. Returns an empty DataFrame if there are no numeric columns.
- **Returns:** one row per numeric column, sorted by skew (most skewed first).
- **Act on it:** high absolute skew (roughly beyond ±1) means a log or power transform will likely help linear models and distance-based methods; tree-based models (random forest, gradient boosting) don't care about skew and can be left alone. A big gap between `mean` and `median` is another sign of skew or outlier influence.

### Validation (these need a spec from you)

These three are different from everything above: they don't infer what's "wrong" automatically — **you** tell them the rules for this specific dataset, and they check the data against those rules. This can't be fully generic (a negative age is invalid; a negative account balance might be completely normal), but the mechanism for checking is reusable across any dataset.

#### `schema_validation(df, expected_schema) -> DataFrame`
- **Checks:** does the dataframe match a dtype contract you define?
- **Expects:** `expected_schema` — a dict of `{column_name: expected_dtype}`, e.g. `{"age": "int64", "income": "float64", "country": "object"}`.
- **Returns:** one row per column (expected or actual), with `status` of `ok`, `dtype_mismatch`, `missing_column` (you expected it, it's not there), or `unexpected_column` (it's there, you didn't account for it).
- **Act on it:** `missing_column` usually means an upstream extract or join dropped a field — treat as a pipeline bug. `dtype_mismatch` (e.g. you expected `float64` but got `object`) is a classic sign that a stray non-numeric value (a typo, a placeholder string like `"N/A"`) is silently corrupting the whole column's type — find and fix that value before modeling. `unexpected_column` isn't necessarily bad, just update your schema once you've confirmed it's intentional.

#### `range_validation(df, rules) -> DataFrame`
- **Checks:** numeric bounds and/or category whitelists, in one pass.
- **Expects:** `rules` — a dict of `{column_name: constraint_dict}`, where each constraint dict can have `"min"`, `"max"`, and/or `"allowed"` (a list of permitted values). Example: `{"age": {"min": 0, "max": 120}, "status": {"allowed": ["active", "closed", "pending"]}}`.
- **Returns:** one row per constraint you defined (a column with both `min` and `max` produces two rows), with `violation_count` and `violation_pct`.
- **Act on it:** any violation count above 0 needs a judgment call: is this bad data (a negative age → almost certainly an error, fix or drop it) or a rule that was wrong (maybe -1 is a legitimate "unknown" sentinel value in this dataset → adjust the rule, not the data). Don't silently clip or drop without checking which case you're in.

#### `cross_field_validation(df, rules) -> DataFrame`
- **Checks:** logical relationships *between* two columns in the same row.
- **Expects:** `rules` — a list of `(col_a, operator, col_b)` tuples. Supported operators: `>=`, `>`, `<=`, `<`, `==`, `!=`. Example: `[("end_date", ">=", "start_date"), ("net_income", "<=", "gross_income")]`. Rows where either column is missing are skipped, not flagged.
- **Returns:** one row per rule, with `violation_count` and `violation_pct`.
- **Act on it:** this is one of the highest-signal checks in the toolkit — a violation here usually means a genuine data entry or pipeline error (an end date before a start date isn't a modeling nuisance, it's a broken record). Pull the violating rows out and inspect them individually rather than aggregating past them.

### Master wrapper

#### `run_data_audit(df, target_col, id_col=None, expected_schema=None, range_rules=None, cross_field_rules=None) -> dict`
- **Checks:** everything above, in one call.
- **Expects:** `target_col` is an existing column name. `id_col` is optional; if supplied, it must name an existing column. `expected_schema`, `range_rules`, `cross_field_rules` are optional — pass them once you know what "valid" means for this dataset, and their reports get added to the output automatically.
- **Returns:** a dict keyed by report name (`"schema"`, `"target"`, `"id"`, `"duplicates"`, `"missingness"`, `"hidden_missingness"`, `"category_normalization"`, `"infinite_values"`, `"cardinality"`, `"constant_features"`, `"numeric_profile"`, plus the three validation reports if you supplied rules for them).
- **Without an ID:** `audit["id"]` is `None`, meaning the identifier report is unavailable. Full-row duplicate checking still runs. The audit does not infer an ID, use the dataframe index as an ID, or generate a key. Check duplicates before creating unique row keys, which would make otherwise identical rows appear different.
- **Act on it:** treat this as your standard "first thing I run on any new dataset" call. Loop through the dict and eyeball each report before writing a single line of feature engineering.

The audit reports findings; it does not clean the dataframe or split it. A misspelled ID column raises an error rather than silently disabling ID checks.

---

## `splitting.py` reference

### Save row identity and train/test membership

```python
from ml_gearbox.splitting import create_split, load_split

result = create_split(
    df,
    id_col=None,
    stratify_col=None,
    group_col=None,
    test_size=0.2,
    random_state=42,
    path="split.parquet",
    overwrite=False,
)
```

`create_split` copies the input dataframe, assigns every row to `train` or `test`, saves the assignments, and returns a dictionary:

| Key | What it contains |
| --- | --- |
| `df` | A copy of the full dataset, with a row key added if needed |
| `id_col` | The row-key column name, such as `transaction_id` or `_row_id` |
| `trainidx` | Zero-based row positions for selecting training rows from this returned dataframe |
| `testidx` | Zero-based row positions for selecting test rows from this returned dataframe |
| `assignments` | The saved membership table, with `row_id` and `split` columns |

The dataframe you pass in is unchanged. Row identity follows these rules:

1. If you supply `id_col`, use that column. Its values must be unique and non-null.
2. Otherwise, reuse an existing `_row_id` column after the same validation.
3. Otherwise, generate unique UUID strings in a new `_row_id` column on the returned copy.

Generated keys belong to this saved dataset. Preserve them in subsequent notebooks; creating keys again from the original dataframe creates different identities. Pandas index labels and row positions are not persistent identities.

| Argument | Purpose |
| --- | --- |
| `id_col` | Identify individual observations when restoring membership |
| `stratify_col` | Keep category proportions approximately similar in train and test; usually a classification target |
| `group_col` | Keep all observations with the same group value together |
| `test_size` | Test fraction or count; defaults to `0.2` |
| `random_state` | Random seed; defaults to `42` |
| `path` | Destination for the membership file |
| `overwrite` | Allow replacement of an existing destination; defaults to `False` |

With neither strategy column, splitting is random. With `stratify_col`, splitting balances the selected categories. With `group_col`, it splits groups; **`test_size` then refers to groups**, so the proportion of test rows can differ. Combining stratification and grouping is unsupported and raises an error.

For example, if a customer has several transactions, `customer_id` repeats and cannot identify each transaction. Use a unique `transaction_id` or generated `_row_id` for identity, and `group_col="customer_id"` to keep each customer's transactions together.

The function rejects missing strategy columns or values, invalid split sizes, too few groups, and class counts that cannot support the requested stratification. Train and test must be nonempty, disjoint, and cover every row. An existing destination raises an error unless you explicitly set `overwrite=True`.

`split.parquet` contains only the mapping:

| row_id | split |
| --- | --- |
| a123 | train |
| b456 | test |

Parquet metadata records the schema version, original ID column and key dtype, strategy and strategy column, requested test size, random seed, and original row count. Feature values and target values are not stored in this file.

### Restore the saved membership

```python
parts = load_split(df, "split.parquet", allow_subset=False)
train_df = parts["train"]
test_df = parts["test"]
```

`load_split` reads the saved mapping and metadata, finds the recorded key column in `df`, and returns training and test copies with the input's feature columns, order, and index preserved. Reordering rows or adding features is fine as long as every row retains its key. It never reruns the random split or regenerates missing keys.

By default, the input must contain exactly the saved set of keys. Use `allow_subset=True` after deliberately filtering out original rows. Every remaining key must still have an assignment; a subset can yield an empty train or test dataframe.

The loader rejects missing, null, duplicate, unknown, or incompatible keys; malformed metadata; duplicate mapping keys; and missing or unsupported split labels. The complete saved mapping must include both `train` and `test`. Equivalent integer dtypes after a Parquet round trip are accepted, but lossy conversions such as changing string ID `"001"` to integer `1` are rejected.

A one-to-many join can duplicate keys and will fail validation. Define how the resulting observations should be identified before restoring a row-level split. Matching row keys restores membership; it does not prove that the source feature or target values have stayed unchanged.

### Two notebooks, explained line by line

Assume `raw_df` already contains your original data and its prediction column is actually named `target`. Replace `"target"` with your own column name, such as `"default"` or `"churn"`.

```python
import pandas as pd
from ml_gearbox.audit import run_data_audit
from ml_gearbox.splitting import create_split, load_split

# Notebook 1
audit = run_data_audit(raw_df, target_col="target")
result = create_split(raw_df, stratify_col="target", path="split.parquet")
result["df"].to_parquet("prepared.parquet", index=False)
train_df = result["df"].iloc[result["trainidx"]].copy()
test_df = result["df"].iloc[result["testidx"]].copy()

# Notebook 2: import pandas and load_split here too.
df = pd.read_parquet("prepared.parquet")
parts = load_split(df, "split.parquet")
train_df, test_df = parts["train"], parts["test"]
```

1. **`audit = run_data_audit(...)` inspects the original data.** It stores the reports in the dictionary `audit`. Look at reports such as `audit["missingness"]` and `audit["duplicates"]`, and address problems before continuing. No rows are cleaned or split by this line. Because no `id_col` was supplied, `audit["id"]` is `None`.
2. **`result = create_split(...)` makes and saves the split.** `stratify_col="target"` requests approximately similar target-class proportions in both parts. The default is about 80% training and 20% test. With no existing ID, it adds `_row_id` to a copy and writes each key's train/test assignment to `split.parquet`. `result` is a dictionary holding the copied dataframe, indices, and assignment information; it is not itself a dataframe.
3. **`result["df"].to_parquet(...)` saves the full dataset with its keys.** `result["df"]` retrieves the dataframe from that dictionary. `to_parquet` writes all of its rows and columns, including the target and `_row_id`, to `prepared.parquet`. `index=False` leaves out pandas' separate index labels; it keeps `_row_id` because that is an ordinary column. This save is what lets another notebook recognize the same observations.
4. **`train_df = result["df"].iloc[result["trainidx"]].copy()` selects training rows.** `result["trainidx"]` retrieves their row positions, for example `[0, 2, 4]`. `.iloc[...]` selects those positions from the returned dataframe. `.copy()` gives you a separate dataframe to work with. The next line does the same for test positions. These indices apply to this dataframe's current order; use `load_split` after reordering rows.

The files serve different purposes:

| File | Contents | Why save it? |
| --- | --- | --- |
| `prepared.parquet` | The complete dataset and its preserved row keys | Let later notebooks load the same observations |
| `split.parquet` | Row keys, train/test labels, and split metadata | Restore each observation's original membership |

In Notebook 2, `pd.read_parquet` loads the complete saved dataset. You can add features or change row order while keeping the key column. `load_split` matches those keys to `split.parquet`, and `parts["train"]` and `parts["test"]` retrieve the resulting dataframes. The final assignment puts those two results into `train_df` and `test_df`. You do not create another split.

---

## `eda.py` reference

### Distributions

#### `target_distribution(df, target) -> DataFrame`
- **Checks:** the same thing as `target_report` in `audit.py` — count and percentage per class of the target.
- **Expects:** `target` is a column name.
- **Returns:** one row per class.
- **Act on it:** same as `target_report` — use it to catch class imbalance early. (These two functions currently duplicate each other; a future cleanup will likely have `eda` call `audit`'s version instead of recomputing it.)

#### `zero_percentage(df, columns) -> DataFrame`
- **Checks:** what fraction of each numeric column is exactly `0` (as opposed to missing).
- **Expects:** a list of numeric column names.
- **Returns:** `feature`, `zero_pct`, only for columns with at least one zero, sorted highest first.
- **Act on it:** a high zero percentage isn't automatically a problem — sometimes zero is a genuine value (zero purchases, zero late payments). But if it's unexpectedly high for a column like "income" or "age", it may actually be a placeholder for missing data that never got converted to `NaN`. Check the data dictionary or source system before treating zeros at face value.

### Outliers

#### `iqr_outlier_summary(df, columns) -> DataFrame`
- **Checks:** outliers per numeric column using the standard IQR rule (outside `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`).
- **Expects:** a list of numeric column names.
- **Returns:** `feature`, `q1`, `q3`, `iqr`, `lower_bound`, `upper_bound`, `outlier_count`, `outlier_pct`, sorted worst first.
- **Act on it:** a handful of outliers (under ~1-2%) is normal — look at a few individually to confirm they're real, not data entry errors, and decide whether to cap/winsorize, transform (log), or leave them (tree-based models are largely outlier-robust; linear/distance-based models are not). A very high outlier percentage usually means the IQR rule doesn't fit this column's distribution (e.g. it's naturally right-skewed, like income) rather than that 20% of your data is broken — pair this with `numeric_profile`'s skew column before reacting.

### Target relationships

#### `categorical_target_rate(df, feature, target) -> DataFrame`
- **Checks:** the target rate (e.g. default rate, churn rate) within each category of a feature.
- **Expects:** `target` should be numeric 0/1 (the function calls `.mean()` on it, which needs a numeric target).
- **Returns:** `feature` (category), `count`, `positives`, `target_rate`, `target_rate_pct`, sorted highest rate first.
- **Act on it:** big spread in target rate across categories → this feature likely has real predictive power, keep it. Also check `count` per category — a category with a dramatic rate but only 3 rows is noise, not signal; consider grouping small categories together.

#### `numeric_target_rate_by_quantile(df, feature, target, bins=10) -> DataFrame`
- **Checks:** the same idea as above, but for a continuous feature — splits it into quantile bins and shows the target rate per bin.
- **Expects:** `target` numeric 0/1. **Important:** the feature column must not contain `inf` values (see Known Quirks below) — run `infinite_value_report` first and clean those up.
- **Returns:** `bin` (the quantile range), `count`, `positives`, `target_rate`, `target_rate_pct`.
- **Act on it:** a target rate that rises or falls monotonically across bins → strong signal, this feature (or a transformed/binned version of it) will likely help a model. A non-monotonic, jagged pattern (e.g. high-low-high-low) → a linear model will miss this relationship even though it's real; a tree-based model or explicit binning will capture it better.

### Correlations

#### `correlation_matrix(df, columns, method="pearson") -> DataFrame`
- **Checks:** pairwise correlation between numeric columns.
- **Expects:** a list of numeric columns; `method` can be `"pearson"` (linear), `"spearman"` (monotonic/rank-based), or `"kendall"`.
- **Returns:** a square correlation matrix.
- **Act on it:** this is usually consumed by `high_correlation_pairs` and `plot_correlation_heatmap` rather than read raw — see below.

#### `high_correlation_pairs(corr_matrix, threshold=0.85) -> DataFrame`
- **Checks:** pulls out just the feature pairs from a correlation matrix that exceed a given absolute correlation.
- **Expects:** the output of `correlation_matrix(...)`.
- **Returns:** `feature_1`, `feature_2`, `correlation`, sorted by absolute strength. Empty DataFrame if nothing crosses the threshold.
- **Act on it:** pairs above ~0.85–0.9 are close to redundant. For linear/logistic regression, this kind of multicollinearity inflates coefficient variance and makes them unstable — drop one feature from each pair, or combine them (e.g. a ratio or PCA component). Tree-based models tolerate this better, but you're still paying for two features' worth of noise for one feature's worth of signal.

### Plots

All plotting functions call `plt.show()` and don't return a figure — they're meant for interactive/notebook use. Each one mirrors a data function above so you get the numbers and the picture together.

#### `plot_numeric_distribution(df, column)`
- **Shows:** a histogram (with KDE) and a boxplot side by side for one numeric column.
- **Use it after:** `numeric_profile` flags a column with high skew, or `iqr_outlier_summary` flags one with a high outlier percentage — this lets you see the shape, not just the number.

#### `plot_categorical_distribution(df, column, top_n=15)`
- **Shows:** a horizontal bar chart of category frequency for a categorical column (capped at the top N categories, default 15, so it stays readable on high-cardinality columns).
- **Use it after:** `cardinality_report` flags a column — visually confirm whether it's a few dominant categories with a long thin tail (a good candidate for "group rare into Other"), or genuinely spread out.

#### `plot_numeric_by_target(df, feature, target)`
- **Shows:** overlapping density curves of a numeric feature, split by target class.
- **Use it after:** you want to eyeball whether a feature separates the two classes at all, before formally checking with `numeric_target_rate_by_quantile`. Curves that barely overlap → strong feature. Curves that sit almost on top of each other → weak feature.

#### `plot_categorical_target_rate(df, feature, target)`
- **Shows:** the same information as `categorical_target_rate`, as a horizontal bar chart.
- **Use it after:** calling `categorical_target_rate` — bars make it much faster to spot which categories are driving risk/behavior than scanning a table.

#### `plot_numeric_target_rate(df, feature, target, bins=10)`
- **Shows:** the same information as `numeric_target_rate_by_quantile`, as a line plot across bins.
- **Use it after:** calling `numeric_target_rate_by_quantile` — the line makes monotonic vs. jagged relationships immediately visible (see the "Act on it" note above on why that distinction matters for model choice). Same `inf`-value caveat applies.

#### `plot_correlation_heatmap(corr_matrix)`
- **Shows:** a heatmap of a correlation matrix, annotated with the actual coefficient values.
- **Use it after:** calling `correlation_matrix` — faster than scanning `high_correlation_pairs` when you want the full picture of how everything relates to everything, not just the pairs above a threshold.

---

## Known quirks

- **`numeric_target_rate_by_quantile` (and its plot) will crash on `inf` values.** `pd.qcut` can't build bin edges around an infinite value. This is a real interaction between the two modules: `infinite_value_report` exists specifically to catch this kind of thing, so make it a habit to run it — and clean up whatever it finds — before reaching for any quantile-based EDA function.
- **`target_distribution` (eda.py) and `target_report` (audit.py) do the same calculation.** Harmless for now, but if you extend one, extend both, or better, consolidate them the next time you touch this code.
- **`schema_validation`'s dtype comparison is exact.** Depending on your pandas version, a plain string column may report as dtype `object` or as a newer string-specific dtype. If you get unexpected `dtype_mismatch` rows on columns that look fine, check what `df[col].dtype` actually prints in your environment and adjust `expected_schema` to match — this isn't a bug, it's the check doing its job of catching a dtype you didn't expect.

## Extending this toolkit

When adding a new function, keep the two existing conventions:
1. **Data functions return a DataFrame (or dict)**, never a plot. Keep plotting in separate `plot_*` functions that consume the data function's output where possible — that's what makes each check independently testable and reusable outside of a notebook.
2. **Document it here in the same format**: what it checks, what it expects, what it returns, and what decision to make from the result. A check nobody knows how to act on isn't pulling its weight in a toolkit like this one.
