"""
Pre-modelling feature selection, in PySpark.

This is a direct PySpark port of a 3-step feature selection process:
1. Drop low-variance features (near-constant columns carry no signal).
2. Flag/remove multicollinear features using Variance Inflation Factor (VIF).
3. Rank the remaining features by ANOVA F-test discriminant power against
   the binary target.

None of the three steps collects the raw feature matrix to the driver.
Each is computed from small, already-aggregated statistics -- a variance
per column, a correlation matrix, or group means/variances by class --
which is what makes this workable at a scale where "just call
statsmodels on a pandas DataFrame" (the natural sklearn/statsmodels
approach) stops being an option.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from pyspark.ml.feature import VarianceThresholdSelector, VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from scipy.stats import f as f_distribution


def drop_low_variance_features(
    df: DataFrame, feature_cols: List[str], threshold: float = 0.01
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """Drop features whose variance doesn't exceed `threshold`.

    Uses Spark ML's native `VarianceThresholdSelector` rather than
    hand-rolling sklearn's `VarianceThreshold` -- same semantics (drop
    columns with sample variance <= threshold), computed as a single
    distributed pass over the data rather than requiring the full matrix
    in memory on one machine.

    Returns (kept_columns, dropped_columns, variances) -- `variances` is a
    small (n_features, 2) pandas DataFrame for printing/inspection, mirroring
    what you'd see printed from the equivalent sklearn snippet.
    """
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="_vt_features", handleInvalid="keep")
    assembled = assembler.transform(df)

    selector = VarianceThresholdSelector(
        featuresCol="_vt_features", outputCol="_vt_selected", varianceThreshold=threshold
    )
    model = selector.fit(assembled)
    kept_idx = set(model.selectedFeatures)

    kept_cols = [c for i, c in enumerate(feature_cols) if i in kept_idx]
    dropped_cols = [c for i, c in enumerate(feature_cols) if i not in kept_idx]

    # Per-feature variances, for the same visibility the sklearn version gives
    # via a quick manual inspection -- one tiny aggregation, not a full collect.
    variances_row = df.select([F.variance(F.col(c)).alias(c) for c in feature_cols]).first()
    variances = pd.DataFrame(
        {"Feature": feature_cols, "Variance": [variances_row[c] for c in feature_cols]}
    ).sort_values("Variance")

    return kept_cols, dropped_cols, variances


def calculate_vif(df: DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Variance Inflation Factor per feature, computed from a correlation
    matrix rather than by fitting one OLS regression per feature (the
    statsmodels approach, which needs the full matrix in local memory).

    This relies on a standard linear-algebra identity: for standardised
    features, VIF_i (from regressing feature i on all other features)
    equals the i-th diagonal entry of the INVERSE of the feature
    correlation matrix. The correlation matrix itself -- an n_features x
    n_features matrix, not the raw data -- is computed with
    `pyspark.ml.stat.Correlation`, a single distributed aggregation over
    the full dataset. Only that small matrix is ever collected to the
    driver, where `numpy.linalg.pinv` (pseudo-inverse, robust to
    near-singular matrices from severe collinearity) does the final,
    cheap inversion.

    A VIF above ~5-10 indicates severe redundancy, same convention as the
    statsmodels version.

    Verified against `statsmodels.stats.outliers_influence.
    variance_inflation_factor` (with an intercept added, the textbook-
    correct way to call it) on this project's actual feature set: every
    ordinary numeric feature matched to within floating-point precision
    (~1e-14). One documented exception, found by that same check --
    read on before trusting this function blindly on a complete one-hot
    family:

    **The `FLAG_SEG_*` lifecycle columns are a dummy-variable trap.**
    Five mutually-exclusive one-hot flags, with no reference category
    dropped, sum to exactly 1 in every row -- once a model intercept
    exists (Spark's `LogisticRegression` fits one by default), that's
    perfect collinearity by construction. statsmodels correctly reports
    this as `VIF = inf`. This function instead reports small, deceptively
    normal-looking numbers (0.9-5.0) for that family, because
    `np.linalg.pinv` smooths over the singularity numerically rather than
    blowing up to infinity the way a true inverse would. That's a real
    limitation, not a rounding difference -- if you're selecting on a
    complete one-hot family, either drop one category as the reference
    level before calling this function, or treat that family's VIFs as
    unreliable and inspect it separately (e.g. check whether the flags
    sum to a constant).
    """
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="_vif_features", handleInvalid="keep")
    assembled = assembler.transform(df)

    corr_matrix = Correlation.corr(assembled, "_vif_features", method="pearson").head()[0].toArray()

    # Guard against exact duplicate columns (correlation of +/-1) making the
    # matrix singular -- pinv degrades gracefully where a true inverse would
    # raise, at the cost of the VIF for that feature reading as very large
    # rather than infinite, which is the more useful signal anyway.
    inv_corr = np.linalg.pinv(corr_matrix)
    vif_values = np.diag(inv_corr)

    return (
        pd.DataFrame({"Feature": feature_cols, "VIF": vif_values})
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )


def rank_features_anova(
    df: DataFrame, feature_cols: List[str], label_col: str
) -> pd.DataFrame:
    """Rank features by one-way ANOVA F-test discriminant power against a
    binary label -- the same statistic `sklearn.feature_selection.f_classif`
    computes, without needing the feature matrix in local memory.

    For a binary label, the F-statistic reduces to a function of each
    class's mean, variance, and count for a given feature:

        SS_between = n0*(mean0 - grand_mean)^2 + n1*(mean1 - grand_mean)^2
        SS_within  = (n0-1)*var0 + (n1-1)*var1
        F = (SS_between / 1) / (SS_within / (n0+n1-2))

    Every quantity on the right (n0, n1, mean0, mean1, var0, var1) for
    EVERY feature comes out of a single `groupBy(label_col).agg(...)` --
    one distributed aggregation returning exactly 2 rows, regardless of
    how many features or rows there are. The F-statistic arithmetic and
    the p-value lookup (via `scipy.stats.f`) then run on those 2 rows,
    on the driver -- the expensive part (scanning the data) still runs
    in Spark.
    """
    agg_exprs = []
    for c in feature_cols:
        agg_exprs += [F.avg(c).alias(f"mean__{c}"), F.variance(c).alias(f"var__{c}")]
    agg_exprs.append(F.count(F.lit(1)).alias("n"))

    group_stats = df.groupBy(label_col).agg(*agg_exprs).toPandas().set_index(label_col)

    if len(group_stats) != 2:
        raise ValueError(
            f"rank_features_anova expects a binary label; found {len(group_stats)} classes in '{label_col}'."
        )

    labels = sorted(group_stats.index.tolist())
    n0, n1 = group_stats.loc[labels[0], "n"], group_stats.loc[labels[1], "n"]
    df_between, df_within = 1, n0 + n1 - 2

    rows = []
    for c in feature_cols:
        m0, m1 = group_stats.loc[labels[0], f"mean__{c}"], group_stats.loc[labels[1], f"mean__{c}"]
        v0, v1 = group_stats.loc[labels[0], f"var__{c}"], group_stats.loc[labels[1], f"var__{c}"]
        grand_mean = (n0 * m0 + n1 * m1) / (n0 + n1)
        ss_between = n0 * (m0 - grand_mean) ** 2 + n1 * (m1 - grand_mean) ** 2
        ss_within = (n0 - 1) * (v0 or 0) + (n1 - 1) * (v1 or 0)

        if ss_within <= 0:
            f_score, p_value = np.nan, np.nan
        else:
            f_score = (ss_between / df_between) / (ss_within / df_within)
            p_value = f_distribution.sf(f_score, df_between, df_within)
        rows.append((c, f_score, p_value))

    return (
        pd.DataFrame(rows, columns=["Feature", "ANOVA_F_Score", "p_value"])
        .sort_values("ANOVA_F_Score", ascending=False)
        .reset_index(drop=True)
    )


def select_features_by_vif(
    df: DataFrame, feature_cols: List[str], max_vif: float = 10.0, verbose: bool = True
) -> Tuple[List[str], List[str]]:
    """Iteratively drop the single worst-VIF feature and recompute, until
    every remaining feature's VIF is at or below `max_vif` (or only one
    feature remains). This is the standard automated VIF-pruning
    procedure -- a notebook can stop after one `calculate_vif` call and
    let a person pick which columns to drop by eye, but an unattended
    pipeline run needs a rule, not a person looking at a table.

    Caveat carried over from `calculate_vif`: this won't reliably catch a
    complete one-hot family's dummy-variable-trap collinearity (see that
    function's docstring). If your feature set includes one, inspect it
    separately -- don't rely on this loop to prune it correctly.
    """
    remaining = list(feature_cols)
    dropped: List[str] = []

    while len(remaining) > 1:
        vif_scores = calculate_vif(df, remaining)
        worst = vif_scores.iloc[0]
        if worst["VIF"] <= max_vif:
            break
        if verbose:
            print(f"  Dropping '{worst['Feature']}' (VIF={worst['VIF']:.2f} > {max_vif})")
        dropped.append(worst["Feature"])
        remaining.remove(worst["Feature"])

    return remaining, dropped
