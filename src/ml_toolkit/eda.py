import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================================
# 1. TARGET & BASIC DISTRIBUTIONS
# ==========================================


def target_distribution(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Returns the absolute and percentage distribution of the target variable."""
    counts = df[target].value_counts(dropna=False).sort_index()
    pcts = df[target].value_counts(normalize=True, dropna=False).sort_index() * 100

    return pd.DataFrame({"count": counts, "percentage": pcts})


def zero_percentage(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Calculates the percentage of exact zero values in numeric features."""
    zeros = df[columns].eq(0).mean() * 100
    return (
        zeros[zeros > 0]
        .sort_values(ascending=False)
        .to_frame("zero_pct")
        .reset_index(names="feature")
    )


# ==========================================
# 2. OUTLIERS & ANOMALIES
# ==========================================


def iqr_outlier_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Calculates IQR bounds and returns the count/percentage of outliers per feature."""
    records = []
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)

        mask = (df[col] < lower) | (df[col] > upper)

        records.append(
            {
                "feature": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": mask.sum(),
                "outlier_pct": mask.mean() * 100,
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values("outlier_pct", ascending=False)
        .reset_index(drop=True)
    )


# ==========================================
# 3. TARGET RELATIONSHIPS (RISK ANALYSIS)
# ==========================================


def categorical_target_rate(
    df: pd.DataFrame, feature: str, target: str
) -> pd.DataFrame:
    """Calculates the positive rate (e.g., default rate) for each category."""
    summary = (
        df.groupby(feature, dropna=False)[target]
        .agg(count="count", positives="sum", target_rate="mean")
        .sort_values("target_rate", ascending=False)
    )

    summary["target_rate_pct"] = summary["target_rate"] * 100
    return summary.reset_index()


def numeric_target_rate_by_quantile(
    df: pd.DataFrame, feature: str, target: str, bins: int = 10
) -> pd.DataFrame:
    """Bins a continuous feature into quantiles and calculates the target rate per bin."""
    temp = df[[feature, target]].dropna().copy()
    temp["bin"] = pd.qcut(temp[feature], q=bins, duplicates="drop")

    result = temp.groupby("bin", observed=True)[target].agg(
        count="count", positives="sum", target_rate="mean"
    )
    result["target_rate_pct"] = result["target_rate"] * 100
    return result.reset_index()


# ==========================================
# 4. CORRELATIONS
# ==========================================


def correlation_matrix(
    df: pd.DataFrame, columns: list[str], method: str = "pearson"
) -> pd.DataFrame:
    """Returns the correlation matrix for the specified numerical columns."""
    return df[columns].corr(method=method)


def high_correlation_pairs(
    corr_matrix: pd.DataFrame, threshold: float = 0.85
) -> pd.DataFrame:
    """Extracts feature pairs that exceed the absolute correlation threshold."""
    pairs = []
    cols = corr_matrix.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) >= threshold:
                pairs.append(
                    {"feature_1": cols[i], "feature_2": cols[j], "correlation": corr}
                )

    if not pairs:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])

    return (
        pd.DataFrame(pairs)
        .sort_values("correlation", key=abs, ascending=False)
        .reset_index(drop=True)
    )


# ==========================================
# 5. VISUALIZATIONS
# ==========================================


def plot_numeric_distribution(df: pd.DataFrame, column: str):
    """Plots a histogram and boxplot side-by-side for a numeric feature."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    sns.histplot(data=df, x=column, kde=True, ax=axes[0])
    axes[0].set_title(f"{column} — Distribution")

    sns.boxplot(data=df, x=column, ax=axes[1])
    axes[1].set_title(f"{column} — Boxplot")

    plt.tight_layout()
    plt.show()


def plot_categorical_distribution(df: pd.DataFrame, column: str, top_n: int = 15):
    """Plots a horizontal bar chart of category frequencies for a categorical feature."""
    counts = df[column].value_counts(dropna=False).head(top_n)

    plt.figure(figsize=(10, max(3, 0.4 * len(counts))))
    sns.barplot(x=counts.values, y=counts.index.astype(str), orient="h")
    plt.title(f"{column} — Category Frequency" + (f" (top {top_n})" if len(counts) == top_n else ""))
    plt.xlabel("Count")
    plt.ylabel(column)
    plt.tight_layout()
    plt.show()


def plot_numeric_by_target(df: pd.DataFrame, feature: str, target: str):
    """Plots the overlapping density distribution of a feature split by the target class."""
    plt.figure(figsize=(10, 5))

    sns.histplot(
        data=df,
        x=feature,
        hue=target,
        stat="density",
        common_norm=False,
        element="step",
        kde=True,
    )

    plt.title(f"{feature} Distribution by {target}")
    plt.show()


def plot_categorical_target_rate(df: pd.DataFrame, feature: str, target: str):
    """Plots the target rate (e.g., default rate) per category as a horizontal bar chart."""
    summary = categorical_target_rate(df, feature, target)

    plt.figure(figsize=(10, max(3, 0.4 * len(summary))))
    sns.barplot(data=summary, x="target_rate_pct", y=feature, orient="h")
    plt.title(f"{target} Rate by {feature}")
    plt.xlabel(f"{target} rate (%)")
    plt.tight_layout()
    plt.show()


def plot_numeric_target_rate(df: pd.DataFrame, feature: str, target: str, bins: int = 10):
    """Plots the target rate across quantile bins of a numeric feature."""
    summary = numeric_target_rate_by_quantile(df, feature, target, bins=bins)

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=summary["bin"].astype(str), y=summary["target_rate_pct"], marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{target} Rate across {feature} Quantile Bins")
    plt.ylabel(f"{target} rate (%)")
    plt.xlabel(feature)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(corr_matrix: pd.DataFrame):
    """Plots a heatmap of a precomputed correlation matrix."""
    plt.figure(figsize=(0.8 * len(corr_matrix.columns) + 2, 0.8 * len(corr_matrix.columns) + 2))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()