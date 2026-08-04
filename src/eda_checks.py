"""
Data-quality and exploratory checks, in PySpark.

This module is the direct port of the EDA additions from the case-study
notebook: referential integrity, outlier/sentinel detection, distribution
comparisons, negative-balance auditing, transaction-history sufficiency,
class balance, and empirically-derived thresholds (replacing hardcoded
magic numbers like "500" or "age 35/60" with values read off the data).

Design note: most of these checks reduce a big Spark DataFrame down to a
handful of numbers or a small contingency table *before* anything leaves
the cluster. The two statistical tests that need SciPy (Kolmogorov-Smirnov,
chi-square) only exist in SciPy, not in PySpark, so for those we collect
just the small, already-aggregated pieces (bucketed counts, a crosstab) to
the driver rather than the raw columns. That keeps the expensive part
(scanning millions of rows) distributed, while still getting a real
p-value out of a driver-only library.

Every function here returns a plain dict, so the results can be dumped to
YAML/JSON as a durable EDA report (see pipelines/train_model.py) instead of
disappearing into notebook cell output.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from scipy.stats import chi2_contingency, ks_2samp


def referential_integrity_check(
    customers: DataFrame, loans: DataFrame, transactions: DataFrame
) -> Dict:
    """Check that every loan/transaction ID exists in the customers table.

    A left-anti join is the Spark-native way to ask "which rows on the
    left have no match on the right" -- it's a single distributed join,
    no collecting required, and scales to however large the tables get.
    """
    customer_ids = customers.select("ID")

    orphaned_loans = loans.join(customer_ids, on="ID", how="left_anti")
    orphaned_transactions = transactions.join(customer_ids, on="ID", how="left_anti")

    n_orphaned_loans = orphaned_loans.select("ID").distinct().count()
    n_orphaned_tx = orphaned_transactions.select("ID").distinct().count()

    return {
        "orphaned_loan_ids": n_orphaned_loans,
        "orphaned_transaction_ids": n_orphaned_tx,
        "passed": n_orphaned_loans == 0 and n_orphaned_tx == 0,
    }


def outlier_and_sentinel_check(
    df: DataFrame, column: str, sentinel_values: Tuple[float, ...] = (9999, 99999, 999999, -1)
) -> Dict:
    """IQR-based outlier bounds plus an explicit sentinel-value scan.

    IQR bounds (Q1 - 1.5*IQR, Q3 + 1.5*IQR) are computed via
    `approxQuantile`, Spark's distributed quantile estimator -- exact
    quantiles need a full sort of the column, which is expensive at
    scale, so the bounded-error approximation is the right tool here.

    The sentinel scan exists because IQR bounds alone won't catch a
    placeholder value that happens to sit inside a wide distribution
    (e.g. an income of 99999 next to genuine six-figure earners). We
    check for it explicitly rather than relying on the outlier bounds
    to surface it by coincidence.
    """
    q1, q3 = df.stat.approxQuantile(column, [0.25, 0.75], 0.01)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    total = df.count()
    outliers = df.filter((F.col(column) < lower_bound) | (F.col(column) > upper_bound)).count()
    sentinel_hits = df.filter(F.col(column).isin(list(sentinel_values))).count()
    min_max = df.select(F.min(column).alias("min"), F.max(column).alias("max")).first()

    return {
        "column": column,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "min_value": min_max["min"],
        "max_value": min_max["max"],
        "n_outliers": outliers,
        "pct_outliers": round(100.0 * outliers / total, 2) if total else 0.0,
        "n_sentinel_hits": sentinel_hits,
    }


def distribution_shift_check(
    df: DataFrame,
    value_column: str,
    group_column: str,
    group_a_value,
    group_b_value,
    max_sample_per_group: int = 20000,
) -> Dict:
    """Two-sample Kolmogorov-Smirnov test comparing `value_column` between
    two groups of `group_column` (e.g. borrowers vs. never-borrowed
    prospects on INCOME).

    `ks_2samp` needs the actual sample values (not just summary
    statistics), and only exists in SciPy. To keep this from pulling an
    unbounded amount of data to the driver, each group is capped at
    `max_sample_per_group` rows via a distributed random sample
    (`DataFrame.sample`) taken *before* the collect -- so the expensive
    scan-and-filter still happens across the cluster, and only a bounded
    sample lands on the driver for the actual statistical test.
    """
    group_a_df = df.filter(F.col(group_column) == group_a_value).select(value_column)
    group_b_df = df.filter(F.col(group_column) == group_b_value).select(value_column)

    def _bounded_sample(spark_df: DataFrame) -> np.ndarray:
        n = spark_df.count()
        if n == 0:
            return np.array([])
        fraction = min(1.0, max_sample_per_group / n)
        pdf = spark_df.sample(withReplacement=False, fraction=fraction, seed=42).toPandas()
        return pdf[value_column].dropna().to_numpy()

    sample_a = _bounded_sample(group_a_df)
    sample_b = _bounded_sample(group_b_df)

    if len(sample_a) == 0 or len(sample_b) == 0:
        return {"error": "One or both groups had no data for this comparison."}

    ks_stat, p_value = ks_2samp(sample_a, sample_b)
    return {
        "test": "two_sample_ks",
        "value_column": value_column,
        "group_column": group_column,
        "group_a": str(group_a_value),
        "group_b": str(group_b_value),
        "n_a_sampled": len(sample_a),
        "n_b_sampled": len(sample_b),
        "ks_statistic": ks_stat,
        "p_value": p_value,
        "significant_at_0_05": bool(p_value < 0.05),
    }


def chi_square_independence_check(df: DataFrame, column_a: str, column_b: str) -> Dict:
    """Chi-square test of independence between two categorical columns
    (e.g. GENDER vs. customer_segment) -- used as part of the bias/
    fairness assessment to check whether segment membership is skewed
    by a protected attribute.

    The crosstab itself (a small category x category count table) is
    built with Spark's own `DataFrame.crosstab`, which is a distributed
    groupBy/pivot under the hood. Only that small table -- not the raw
    rows -- is collected to the driver for the chi-square calculation,
    since `chi2_contingency` only exists in SciPy.
    """
    crosstab = df.crosstab(column_a, column_b).toPandas()
    contingency = crosstab.drop(columns=crosstab.columns[0]).to_numpy()

    chi2, p_value, dof, _ = chi2_contingency(contingency)
    return {
        "test": "chi_square_independence",
        "column_a": column_a,
        "column_b": column_b,
        "chi2_statistic": chi2,
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "significant_at_0_05": bool(p_value < 0.05),
    }


def negative_balance_report(transactions: DataFrame) -> Dict:
    """Summarise how often, and how deeply, balances go negative.

    This doesn't decide whether a negative balance is a valid overdraft
    state or a data artefact -- that's a judgement call for whoever reads
    the report -- it just surfaces the scale of the phenomenon so that
    judgement call can be made with evidence instead of assumption.
    """
    total_rows = transactions.count()
    negative = transactions.filter(F.col("BALANCE") < 0)
    n_negative_rows = negative.count()
    n_customers_ever_negative = negative.select("ID").distinct().count()
    total_customers = transactions.select("ID").distinct().count()

    stats = negative.select(
        F.min("BALANCE").alias("min_balance"), F.avg("BALANCE").alias("avg_balance")
    ).first()

    return {
        "n_negative_rows": n_negative_rows,
        "pct_negative_rows": round(100.0 * n_negative_rows / total_rows, 2) if total_rows else 0.0,
        "n_customers_ever_negative": n_customers_ever_negative,
        "pct_customers_ever_negative": (
            round(100.0 * n_customers_ever_negative / total_customers, 2) if total_customers else 0.0
        ),
        "max_overdraft_depth": stats["min_balance"] if stats else None,
        "avg_overdraft_depth": stats["avg_balance"] if stats else None,
    }


def transaction_history_report(
    customer_features: DataFrame, low_history_percentile: float = 0.05
) -> Dict:
    """Check how much transaction history each customer actually has, and
    flag customers below a low-history percentile.

    Aggregate features computed from very few transactions (e.g. an
    average balance based on 2 data points) are much noisier than the
    same feature computed from hundreds of transactions. Rather than
    silently trusting every customer's aggregates equally, this derives
    a data-driven "insufficient history" floor and returns it so the
    caller can add a `FLAG_insufficient_history` feature or otherwise
    treat these customers with more caution downstream.
    """
    threshold = customer_features.stat.approxQuantile(
        "total_transactions_count", [low_history_percentile], 0.01
    )[0]
    n_flagged = customer_features.filter(
        F.col("total_transactions_count") <= threshold
    ).count()
    total = customer_features.count()

    return {
        "low_history_percentile": low_history_percentile,
        "low_history_threshold": threshold,
        "n_customers_flagged": n_flagged,
        "pct_customers_flagged": round(100.0 * n_flagged / total, 2) if total else 0.0,
    }


def class_balance_report(loans: DataFrame, label_column: str = "OUTCOME") -> Dict:
    """Report the raw class balance of the historical loan outcome.

    This is the real supervised-learning target (see features.py), so its
    balance belongs in the EDA report up front -- not discovered
    incidentally while building the train/test split much later.
    """
    total = loans.count()
    counts = loans.groupBy(label_column).count().toPandas()
    counts["pct"] = (100.0 * counts["count"] / total).round(2) if total else 0.0
    return {
        "label_column": label_column,
        "total_rows": total,
        "counts": counts.set_index(label_column)["count"].to_dict(),
        "percentages": counts.set_index(label_column)["pct"].to_dict(),
    }


def derive_empirical_thresholds(
    transactions: DataFrame, customer_features: DataFrame
) -> Dict:
    """Derive feature-engineering thresholds from the data itself, instead
    of hardcoding round numbers.

    Two thresholds used elsewhere in `features.py` are derived here:

    - low_balance_threshold: the original notebook used a hardcoded
      "500" to define a "low balance" event. Here we instead take the
      10th percentile of *positive* balances (excluding already-negative
      accounts, which would pull the percentile down for the wrong
      reason) -- so "low balance" means "bottom decile of what a
      typical, non-overdrawn account looks like" for *this* population,
      not an arbitrary constant that may not fit a different portfolio.
    - lifecycle age tertiles: replacing the hardcoded 35/60 age cutoffs
      with the 33rd/66th percentiles of the actual AGE distribution, so
      the three lifecycle segments are always roughly equal-sized
      regardless of how this customer base's age profile shifts over
      time.
    """
    active_balances = transactions.filter(F.col("BALANCE") > 0)
    low_balance_threshold = active_balances.stat.approxQuantile("BALANCE", [0.10], 0.01)[0]

    age_q33, age_q66 = customer_features.stat.approxQuantile("AGE", [0.33, 0.66], 0.01)

    return {
        "low_balance_threshold": low_balance_threshold,
        "age_tertile_33": age_q33,
        "age_tertile_66": age_q66,
    }


def population_stability_index(
    baseline: DataFrame,
    current: DataFrame,
    column: str,
    num_bins: int = 10,
) -> Dict:
    """Population Stability Index (PSI) between a baseline (e.g. training
    set) and a current (e.g. latest production batch) distribution of one
    numeric column. This is the monitoring-time counterpart to the
    distribution checks above: those ask "are these two groups different
    right now", this asks "has this single feature's distribution drifted
    since training".

    Bin edges are derived from the *baseline* distribution's quantiles
    (via `approxQuantile`, distributed) so that bins reflect where the
    training data actually sits; both distributions are then histogrammed
    into those same bins and compared. Only the bin *counts* -- a handful
    of numbers per distribution -- are collected to the driver; the
    binning and counting themselves run as Spark aggregations.

    Interpretation follows the standard industry convention:
      PSI < 0.10            -> no meaningful shift
      0.10 <= PSI <= 0.25    -> moderate shift, monitor
      PSI > 0.25             -> significant shift, consider retraining
    """
    quantiles = [i / num_bins for i in range(num_bins + 1)]
    edges = baseline.stat.approxQuantile(column, quantiles, 0.01)
    # Guard against duplicate edges (e.g. a heavily repeated value) and
    # make the outer edges open-ended so no value falls outside the bins.
    edges = sorted(set(edges))
    if len(edges) < 2:
        return {"error": f"Not enough distinct values in '{column}' to bin."}
    edges[0] = float("-inf")
    edges[-1] = float("inf")

    def _bin_counts(df: DataFrame) -> np.ndarray:
        # Assign each row to a bin index by walking the quantile edges from
        # the top down: start by assuming the last bin, then override with
        # an earlier bin wherever the value falls below that edge. This
        # gives an ordinary CASE WHEN expression under the hood (one
        # `when` per edge), which Catalyst optimises into a single pass --
        # no Python UDF, no per-row driver round-trip.
        bin_expr = F.lit(len(edges) - 2)
        for i in range(len(edges) - 2, -1, -1):
            bin_expr = F.when(F.col(column) < edges[i + 1], F.lit(i)).otherwise(bin_expr)

        counts_pdf = df.select(bin_expr.alias("bin")).groupBy("bin").count().toPandas()
        counts = np.zeros(len(edges) - 1)
        for _, row in counts_pdf.iterrows():
            counts[int(row["bin"])] = row["count"]
        return counts

    baseline_counts = _bin_counts(baseline)
    current_counts = _bin_counts(current)

    baseline_pct = baseline_counts / max(baseline_counts.sum(), 1)
    current_pct = current_counts / max(current_counts.sum(), 1)

    # Laplace-style smoothing so empty bins don't produce log(0) / division
    # by zero -- a standard adjustment for PSI in industry practice.
    baseline_pct = np.where(baseline_pct == 0, 1e-4, baseline_pct)
    current_pct = np.where(current_pct == 0, 1e-4, current_pct)

    psi = float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))

    if psi < 0.10:
        stability = "stable"
    elif psi <= 0.25:
        stability = "moderate_shift"
    else:
        stability = "significant_shift"

    return {"column": column, "psi": psi, "stability": stability, "num_bins": len(edges) - 1}


def run_full_eda_report(
    customers: DataFrame,
    loans: DataFrame,
    transactions: DataFrame,
    customer_features: DataFrame,
) -> Dict:
    """Run every check in this module and assemble one EDA report dict.

    This is what `pipelines/train_model.py` calls; it's also directly
    reusable from a notebook if you want to inspect one check at a time
    interactively -- each underlying function still works standalone.
    """
    report: Dict = {}

    report["referential_integrity"] = referential_integrity_check(customers, loans, transactions)
    report["income_outliers"] = outlier_and_sentinel_check(customers, "INCOME")
    report["loan_amount_outliers"] = outlier_and_sentinel_check(loans, "AMOUNT")
    report["negative_balance"] = negative_balance_report(transactions)
    report["transaction_history"] = transaction_history_report(customer_features)
    report["class_balance"] = class_balance_report(loans)
    report["empirical_thresholds"] = derive_empirical_thresholds(transactions, customer_features)

    report["income_distribution_shift"] = distribution_shift_check(
        customer_features,
        value_column="INCOME",
        group_column="customer_segment",
        group_a_value="Existing Borrower",
        group_b_value="Deposit-Only Prospect",
    )
    report["gender_segment_independence"] = chi_square_independence_check(
        customer_features, "GENDER", "customer_segment"
    )

    return report

def compute_kpi_correlation_matrix(df: DataFrame, kpi_cols: List[str]) -> pd.DataFrame:
    """Correlation matrix over the named KPI columns, computed with
    pyspark.ml.stat.Correlation (one distributed pass) -- same approach
    as feature_selection.calculate_vif, just returning the raw matrix
    instead of inverting it. Only the small (n_kpi x n_kpi) result is
    collected to the driver.
    """
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.stat import Correlation

    assembled = VectorAssembler(inputCols=list(kpi_cols), outputCol="_kpi_vec").transform(df)
    corr_matrix = Correlation.corr(assembled, "_kpi_vec", method="pearson").head()[0].toArray()
    return pd.DataFrame(corr_matrix, index=kpi_cols, columns=kpi_cols)