"""
Plotting helpers, in matplotlib/seaborn.

Every function here follows the same discipline as `eda_checks.py`: the
expensive part (filtering, aggregating, sampling) happens in Spark, and
only a small, already-reduced pandas DataFrame is ever collected to the
driver for the actual plotting. No function in this module calls
`.collect()` or `.toPandas()` on a full, un-aggregated Spark DataFrame.

This module exists because a notebook full of `.show()` calls -- however
correct -- doesn't satisfy a brief that explicitly asks for
"exploratory analytics and visualization". Text tables and charts answer
different questions; this module is what makes the charts possible
without undoing the "stay distributed" discipline the rest of the project
follows.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# A small, consistent palette used across every chart in the notebook, so
# figures read as one coherent report rather than a grab-bag of defaults.
NAVY = "#1E2761"
ICE = "#8FA8D9"
WARN = "#C24914"
OK = "#2C6E49"
MUTED = "#5B6472"

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "#D7DEEA"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "#EEF1F6"
plt.rcParams["grid.linewidth"] = 0.8
plt.rcParams["font.size"] = 11


def plot_category_counts(
    df: DataFrame, column: str, title: str, color: str = NAVY, ax=None
):
    """Bar chart of value counts for a categorical column.

    The groupBy/count aggregation runs in Spark; only the resulting
    (category, count) pairs -- at most a handful of rows -- are collected
    to the driver via `toPandas()`.
    """
    counts = df.groupBy(column).count().orderBy(F.desc("count")).toPandas()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts[column].astype(str), counts["count"], color=color)
    ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY)
    ax.set_ylabel("Customers")
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    return ax


def plot_bounded_histogram(
    df: DataFrame,
    column: str,
    title: str,
    bins: int = 30,
    max_sample: int = 20000,
    color: str = NAVY,
    ax=None,
    vlines: Optional[Dict[str, float]] = None,
):
    """Histogram of a numeric column, computed from a bounded random sample.

    Matplotlib needs actual values to bin, not just summary statistics, so
    something has to be collected -- but collecting an entire production-
    scale column just to draw a picture defeats the point of using Spark
    in the first place. `DataFrame.sample()` draws a bounded fraction
    *before* the collect, the same pattern used in
    `eda_checks.distribution_shift_check`.
    """
    n = df.count()
    fraction = min(1.0, max_sample / n) if n else 1.0
    values = df.select(column).sample(withReplacement=False, fraction=fraction, seed=42).toPandas()[column].dropna()

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=bins, color=color, alpha=0.85, edgecolor="white")
    ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xlabel(column)
    ax.set_ylabel("Count" + (f" (n={len(values)} sampled)" if fraction < 1.0 else ""))
    if vlines:
        for label, x in vlines.items():
            ax.axvline(x, color=WARN, linestyle="--", linewidth=1.5)
            ax.text(x, ax.get_ylim()[1] * 0.95, f" {label}", color=WARN, fontsize=9, va="top")
    return ax


def plot_grouped_histogram(
    df: DataFrame,
    value_column: str,
    group_column: str,
    group_values: List[str],
    title: str,
    bins: int = 30,
    max_sample_per_group: int = 10000,
    colors: Optional[List[str]] = None,
    ax=None,
):
    """Overlaid histograms of `value_column` for each value in
    `group_values` -- e.g. INCOME for 'Existing Borrower' vs.
    'Deposit-Only Prospect'. The visual companion to
    `eda_checks.distribution_shift_check`'s KS-test: this shows *what*
    the distributions look like; the KS-test says *whether* the
    difference is statistically real.
    """
    colors = colors or [NAVY, WARN, OK]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    for i, gv in enumerate(group_values):
        subset = df.filter(F.col(group_column) == gv).select(value_column)
        n = subset.count()
        if n == 0:
            continue
        fraction = min(1.0, max_sample_per_group / n)
        values = subset.sample(withReplacement=False, fraction=fraction, seed=42).toPandas()[value_column].dropna()
        ax.hist(values, bins=bins, alpha=0.55, label=str(gv), color=colors[i % len(colors)], edgecolor="white")
    ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xlabel(value_column)
    ax.set_ylabel("Count (sampled)")
    ax.legend()
    return ax


def plot_roc_curve(predictions: DataFrame, label_col: str = "TARGET_propensity", ax=None):
    """ROC curve for the test-set predictions.

    The held-out test set is already a bounded, deliberately-sized split
    (not the full dataset), so collecting it in full to the driver here is
    consistent with the rest of the project's "only collect what's already
    small" rule -- this isn't an exception to it.
    """
    from pyspark.ml.functions import vector_to_array
    from sklearn.metrics import roc_curve, auc as sk_auc

    pdf = (
        predictions.withColumn("prob_array", vector_to_array(F.col("probability")))
        .select(label_col, F.col("prob_array").getItem(1).alias("score"))
        .toPandas()
    )
    fpr, tpr, _ = roc_curve(pdf[label_col], pdf["score"])
    roc_auc = sk_auc(fpr, tpr)

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color=NAVY, linewidth=2.2, label=f"Model (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve \u2014 Test Set", fontsize=13, fontweight="bold", color=NAVY)
    ax.legend(loc="lower right")
    return ax


def plot_coefficients(pipeline_model, feature_columns: List[str], top_n: int = 15, ax=None):
    """Horizontal bar chart of the fitted logistic regression's
    coefficients, sorted by absolute magnitude.

    Coefficients live on the model object itself (already on the driver
    once the pipeline is fit) -- nothing to collect from Spark here.
    """
    lr_model = pipeline_model.stages[-1]
    coefs = lr_model.coefficients.toArray()
    pdf = pd.DataFrame({"feature": feature_columns, "coefficient": coefs})
    pdf["abs_coef"] = pdf["coefficient"].abs()
    pdf = pdf.sort_values("abs_coef", ascending=False).head(top_n).sort_values("coefficient")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(pdf))))
    colors = [OK if c > 0 else WARN for c in pdf["coefficient"]]
    ax.barh(pdf["feature"], pdf["coefficient"], color=colors)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_title(f"Top {len(pdf)} Coefficients by Magnitude", fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xlabel("Standardised coefficient (green = pushes toward TakeUp)")
    return ax


def plot_fairness_rates(fairness: Dict, ax=None):
    """Two-panel bar chart of predicted approval rate by gender and by age
    band, straight from `modeling.fairness_report`'s output -- already a
    small dict, nothing further to collect.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    gender_rates = fairness["approval_rate_by_gender"]
    axes[0].bar(list(gender_rates.keys()), list(gender_rates.values()), color=[NAVY, ICE, MUTED])
    axes[0].set_title("Predicted Approval Rate by Gender", fontsize=12, fontweight="bold", color=NAVY)
    axes[0].set_ylabel("Predicted approval rate")
    axes[0].set_ylim(0, 1)

    age_rates = fairness["approval_rate_by_age_band"]
    axes[1].bar(list(age_rates.keys()), list(age_rates.values()), color=NAVY)
    axes[1].set_title("Predicted Approval Rate by Age Band", fontsize=12, fontweight="bold", color=NAVY)
    axes[1].set_ylabel("Predicted approval rate")
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    return fig, axes


def plot_prospect_scores(prospect_scores_pdf: pd.DataFrame, ax=None):
    """Boxplot of the model's extrapolated propensity score, grouped by
    the Step-5 business-rule quadrant -- lets you see at a glance whether
    the model's ranking broadly agrees with the rule, or diverges from it
    for a particular segment. Takes a pandas DataFrame directly (already
    collected by the caller, e.g. via a bounded `.limit()` or because it's
    the small `prospect_scores` output already written to disk).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    order = sorted(prospect_scores_pdf["KPI_action_quadrant"].unique())
    data = [
        prospect_scores_pdf.loc[prospect_scores_pdf["KPI_action_quadrant"] == q, "model_propensity_score"]
        for q in order
    ]
    bp = ax.boxplot(data, labels=order, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(ICE)
        patch.set_edgecolor(NAVY)
    ax.set_title(
        "Model Score vs. Business-Rule Quadrant (Never-Borrowed Prospects)",
        fontsize=13, fontweight="bold", color=NAVY,
    )
    ax.set_ylabel("Model propensity score (unvalidated extrapolation)")
    ax.tick_params(axis="x", rotation=15)
    return ax


def plot_monthly_trend(monthly_pdf: pd.DataFrame, x_col: str, y_col: str, title: str, ax=None):
    """Line chart of a metric over calendar month -- e.g. average burn
    rate per month across the whole customer base. Takes a pandas
    DataFrame directly: monthly aggregates are already small (a handful
    of rows, one per calendar month), so the caller collects once via
    `.toPandas()` and this function only draws it.
    """
    pdf = monthly_pdf.sort_values(x_col)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    ax.plot(pdf[x_col], pdf[y_col], color=NAVY, marker="o", linewidth=2)
    ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.tick_params(axis="x", rotation=40)
    return ax


def plot_top_n_bar(
    pdf: pd.DataFrame, label_col: str, value_col: str, title: str, n: int = 15, color: str = WARN, ax=None
):
    """Horizontal bar chart of the top-N rows by `value_col` -- e.g. the
    customers with the highest average burn rate. Takes a pandas
    DataFrame directly (the caller is expected to have already done the
    Spark-side `.orderBy(...).limit(n)` before collecting, so at most
    `n` rows ever reach this function).
    """
    top = pdf.sort_values(value_col, ascending=False).head(n).sort_values(value_col)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(top))))
    ax.barh(top[label_col].astype(str), top[value_col], color=color)
    ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xlabel(value_col)
    return ax


def plot_feature_selection_scores(discriminant_scores: pd.DataFrame, title: str = "Pre-Modelling Feature Selection: Discriminant Power Ranking", ax=None):
    """Horizontal bar chart of ANOVA F-scores per feature, from
    `feature_selection.rank_features_anova`'s output -- already a small
    pandas DataFrame, nothing further to collect. Colour scales with
    score (darker = more discriminant power), the same visual intent as
    the original seaborn `palette='viridis'` version.
    """
    pdf = discriminant_scores.sort_values("ANOVA_F_Score", ascending=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(pdf))))
    norm_scores = (pdf["ANOVA_F_Score"] - pdf["ANOVA_F_Score"].min()) / (
        pdf["ANOVA_F_Score"].max() - pdf["ANOVA_F_Score"].min() + 1e-9
    )
    colors = plt.cm.viridis(norm_scores)
    ax.barh(pdf["Feature"], pdf["ANOVA_F_Score"], color=colors)
    ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xlabel("ANOVA F-Statistic Score (higher = more discriminant)")
    ax.set_ylabel("Engineered feature candidates")
    return ax


def plot_variance_scores(variances: pd.DataFrame, threshold: float = None, title: str = "Feature Variance (Low-Variance Screen)", ax=None):
    """Horizontal bar chart of per-feature variance, from
    `feature_selection.drop_low_variance_features`'s `variances` output.
    A vertical line marks the drop threshold, if given, so it's visually
    obvious which bars were removed and why.
    """
    pdf = variances.sort_values("Variance", ascending=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(pdf))))
    colors = [WARN if threshold is not None and v <= threshold else NAVY for v in pdf["Variance"]]
    ax.barh(pdf["Feature"], pdf["Variance"], color=colors)
    if threshold is not None:
        ax.axvline(threshold, color=WARN, linestyle="--", linewidth=1.5)
        ax.text(threshold, len(pdf) - 0.5, f"  threshold = {threshold}", color=WARN, fontsize=9, va="top")
    ax.set_xscale("symlog")
    ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xlabel("Variance (log scale) -- red bars fall at or below the drop threshold")
    return ax


def plot_vif_scores(vif_scores: pd.DataFrame, max_vif: float = 10.0, title: str = "Variance Inflation Factor (Multicollinearity Screen)", ax=None):
    """Horizontal bar chart of per-feature VIF, from
    `feature_selection.calculate_vif`'s output. A vertical line marks the
    severe-redundancy threshold (default 10) so it's visually obvious
    which features are flagged.

    Carries the same caveat as `calculate_vif` itself: a complete one-hot
    family (e.g. FLAG_SEG_*) can show deceptively low VIF here due to the
    pseudo-inverse smoothing over that specific kind of singularity --
    see that function's docstring before reading a clean bar for that
    family as "no collinearity."
    """
    pdf = vif_scores.sort_values("VIF", ascending=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(pdf))))
    colors = [WARN if v > max_vif else NAVY for v in pdf["VIF"]]
    ax.barh(pdf["Feature"], pdf["VIF"], color=colors)
    ax.axvline(max_vif, color=WARN, linestyle="--", linewidth=1.5)
    ax.text(max_vif, len(pdf) - 0.5, f"  severe redundancy > {max_vif}", color=WARN, fontsize=9, va="top")
    ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xlabel("Variance Inflation Factor")
    return ax

def plot_kpi_correlation_heatmap(corr_df: pd.DataFrame, title: str = "KPI Feature Correlation Matrix", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr_df.values, cmap="viridis", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=90)
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index)
    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            ax.text(j, i, f"{corr_df.values[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax



def plot_confusion_matrix(
    cm: dict,
    labels: tuple = ("Block (0)", "Target (1)"),
    business_labels: dict = None,
    title: str = "Operational Confusion Matrix & Error Mapping",
    ax=None,
):
    """2x2 confusion matrix with business-friendly labels per quadrant
    (e.g. 'Wasted Promo' for a false positive) -- takes the small dict
    from modeling.compute_confusion_matrix directly, nothing to collect
    here.
    """
    business_labels = business_labels or {
        "true_negative": "Correctly Blocked",
        "false_positive": "Wasted Promo",
        "false_negative": "Missed Target",
        "true_positive": "Correct Conversion",
    }
    matrix = np.array([
        [cm["true_negative"], cm["false_positive"]],
        [cm["false_negative"], cm["true_positive"]],
    ])
    names = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
    biz = [
        [business_labels["true_negative"], business_labels["false_positive"]],
        [business_labels["false_negative"], business_labels["true_positive"]],
    ]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            text_color = "white" if matrix[i, j] > matrix.max() * 0.5 else NAVY
            ax.text(j, i, f"{names[i][j]}\n{matrix[i, j]}\n({biz[i][j]})",
                    ha="center", va="center", fontsize=11, fontweight="bold", color=text_color)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"Predicted {labels[0]}", f"Predicted {labels[1]}"])
    ax.set_yticks([0, 1]); ax.set_yticklabels([f"Actual {labels[0]}", f"Actual {labels[1]}"])
    ax.set_xlabel("Model Classification Prediction")
    ax.set_ylabel("True Portfolio Status")
    ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY)
    return ax