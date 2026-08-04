"""
Feature engineering, in PySpark.

This module builds two tables from the cleaned raw data:

1. `build_customer_features` -- one row per customer, aggregating all of
   their transaction and loan history into the financial-health KPIs
   defined in the case study's assumptions (net cash flow, balance
   volatility, burn rate, credit capacity, etc.), plus the
   `customer_segment` flag (Existing Borrower vs. Deposit-Only Prospect)
   and the `KPI_action_quadrant` business-rule heuristic used later to
   score never-borrowed prospects.

2. `build_loan_level_features` -- one row per *historical loan
   application*, with point-in-time-correct features (only using
   information from applications strictly before the one being scored).
   This is the table the model is actually trained on -- see the
   "why not customer-level" note on that function for the reasoning.

Everything here is a straight, documented port of the case-study
notebook's logic; nothing new was invented, but every KPI now carries a
comment on what it means and why it's computed the way it is, since that
context lived only in scattered notebook markdown cells before.
"""

from __future__ import annotations

from typing import Dict, List

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ---------------------------------------------------------------------------
# Customer-level feature table
# ---------------------------------------------------------------------------

def _transaction_aggregates(transactions: DataFrame) -> DataFrame:
    """One row per customer: basic transaction-derived statistics.

    avg_balance / balance_std / total_transactions_count feed directly
    into several KPIs below and are also used by eda_checks to flag
    customers with too little history to trust their aggregates.
    """
    return transactions.groupBy("ID").agg(
        F.avg("BALANCE").alias("avg_balance"),
        F.stddev("BALANCE").alias("balance_std"),
        F.count(F.lit(1)).alias("total_transactions_count"),
    )


def _loan_aggregates(loans: DataFrame) -> DataFrame:
    """One row per customer: full-history loan aggregates.

    NOTE: these full-history aggregates (total_loans_taken,
    successful_loans, max_loan_amount, ...) are appropriate for
    describing a customer's overall profile (used in EDA, in the
    `customer_segment` flag, and in the Step-5 business-rule heuristic
    for scoring prospects). They are deliberately NOT used as predictors
    when modelling an individual loan application's outcome, because a
    customer's own current application is one of the loans being
    aggregated -- using "successful_loans / total_loans_taken" as a
    feature for that same application would leak its own label into the
    input. See `build_loan_level_features` for the point-in-time-correct
    version used for modelling.
    """
    return loans.groupBy("ID").agg(
        F.count(F.lit(1)).alias("total_loans_taken"),
        F.sum(F.when(F.col("OUTCOME") == "TakeUp", 1).otherwise(0)).alias("successful_loans"),
        F.max("AMOUNT").alias("max_loan_amount"),
        F.max("DATE").alias("last_loan_date"),
    )


def _net_cash_flow(transactions: DataFrame) -> DataFrame:
    """Average monthly net cash flow per customer (Financial Health
    Assumption: "a steady gap between monthly Income and transaction
    Amount implies positive cash flow and capacity to repay").

    Computed as: sum(AMOUNT) per customer per calendar month, then
    averaged across months. Positive AMOUNT = inflow, negative = outflow,
    per the data dictionary, so a plain monthly sum already nets the two.
    """
    monthly = transactions.withColumn("year_month", F.date_format("DATE", "yyyy-MM")).groupBy(
        "ID", "year_month"
    ).agg(F.sum("AMOUNT").alias("monthly_net_flow"))
    return monthly.groupBy("ID").agg(F.avg("monthly_net_flow").alias("KPI_net_cash_flow"))


def compute_burn_rate_detail(transactions: DataFrame) -> DataFrame:
    """Per customer-month burn rate: how many currency units per day a
    customer's balance drops after it peaks each month (Financial Health
    Assumption: "identify how fast the Balance drops after Income is
    deposited").

    Per customer-month: find the peak balance (a proxy for "just got
    paid"), then the lowest balance that occurs on or after that peak,
    and express the drop as an average amount lost per day between the
    two. This needs to look at each customer-month's balance *sequence*
    in DATE order, which is exactly what a window function ordered by
    DATE is for -- no Python UDF or driver-side loop required.

    This is a public, standalone function (rather than inlined into the
    per-customer KPI below) because it's reused for two different
    purposes that both need the same customer-month grain: the
    per-customer average (`KPI_burn_rate_per_day`, via
    `build_customer_features`) and the whole-portfolio monthly trend
    (`compute_monthly_burn_rate_trend`, for the time-series chart). Each
    row here is one customer's burn rate for one calendar month.
    """
    monthly = transactions.withColumn("year_month", F.date_format("DATE", "yyyy-MM"))

    peak_window = Window.partitionBy("ID", "year_month").orderBy(F.col("BALANCE").desc())
    peak = (
        monthly.withColumn("rn", F.row_number().over(peak_window))
        .filter(F.col("rn") == 1)
        .select(
            "ID",
            "year_month",
            F.col("DATE").alias("peak_date"),
            F.col("BALANCE").alias("peak_balance"),
        )
    )

    with_peak = monthly.join(peak, on=["ID", "year_month"], how="inner").filter(
        F.col("DATE") >= F.col("peak_date")
    )

    trough_window = Window.partitionBy("ID", "year_month").orderBy(F.col("BALANCE").asc())
    trough = (
        with_peak.withColumn("rn", F.row_number().over(trough_window))
        .filter(F.col("rn") == 1)
        .select(
            "ID",
            "year_month",
            "peak_date",
            "peak_balance",
            F.col("DATE").alias("trough_date"),
            F.col("BALANCE").alias("trough_balance"),
        )
    )

    return trough.withColumn(
        "days_between", F.datediff(F.col("trough_date"), F.col("peak_date"))
    ).filter(F.col("days_between") > 0).withColumn(
        "burn_rate_per_day",
        (F.col("peak_balance") - F.col("trough_balance")) / F.col("days_between"),
    )


def _burn_rate(transactions: DataFrame) -> DataFrame:
    """Per-customer average burn rate -- thin wrapper around
    `compute_burn_rate_detail`, aggregated up from customer-month to
    customer grain for use as a KPI in `build_customer_features`.
    """
    detail = compute_burn_rate_detail(transactions)
    return detail.groupBy("ID").agg(F.avg("burn_rate_per_day").alias("KPI_burn_rate_per_day"))


def compute_monthly_burn_rate_trend(transactions: DataFrame) -> DataFrame:
    """Whole-portfolio average burn rate by calendar month -- the
    time-series view that answers "is the customer base burning through
    money faster than it used to," as opposed to `KPI_burn_rate_per_day`,
    which only answers "how does this one customer compare to others
    right now."

    This was in the original pandas prototype (a line chart of
    `burn_rate_summary` by month) but was dropped when the notebook was
    first ported to PySpark -- reinstated here on the same
    `compute_burn_rate_detail` grain, just aggregated by month instead of
    by customer.
    """
    detail = compute_burn_rate_detail(transactions)
    return (
        detail.groupBy("year_month")
        .agg(F.avg("burn_rate_per_day").alias("avg_burn_rate_per_day"))
        .orderBy("year_month")
    )


def _low_balance_frequency(transactions: DataFrame, low_balance_threshold: float) -> DataFrame:
    """Count of transactions where BALANCE fell below `low_balance_threshold`
    (Financial Health Assumption: "frequent low or near-zero Balance
    milestones assume immediate cash needs but higher risk").

    `low_balance_threshold` is expected to come from
    `eda_checks.derive_empirical_thresholds` (10th percentile of positive
    balances) rather than a hardcoded constant -- see that function's
    docstring for why.
    """
    return transactions.withColumn(
        "is_low_balance", (F.col("BALANCE") < F.lit(low_balance_threshold)).cast("int")
    ).groupBy("ID").agg(F.sum("is_low_balance").alias("KPI_low_balance_frequency"))


def _lifecycle_segment_flags(
    customer_features: DataFrame, age_q33: float, age_q66: float
) -> DataFrame:
    """One-hot lifecycle segment flags from AGE x INCOME quadrants (Life
    Stage & Needs Assumption: certain age groups paired with income level
    suggest different credit needs -- e.g. young/low-income prospects for
    micro-loans vs. mid-age/high-income prospects for mortgages).

    Age cutoffs are the empirical tertiles (`age_q33`/`age_q66`) rather
    than hardcoded 35/60, so the three age bands are roughly equal-sized
    for this population instead of arbitrary round numbers.
    """
    income_median_row = customer_features.approxQuantile("INCOME", [0.5], 0.01)
    income_median = income_median_row[0] if income_median_row else 0.0

    is_young = F.col("AGE") <= age_q33
    is_midage = (F.col("AGE") > age_q33) & (F.col("AGE") <= age_q66)
    is_senior = F.col("AGE") > age_q66
    is_high_income = F.col("INCOME") > income_median

    segment = (
        F.when(is_young & ~is_high_income, "Young_LowIncome")
        .when(is_midage & is_high_income, "MidAge_HighIncome")
        .when(is_senior, "Senior_Stable")
        .when(is_young & is_high_income, "Young_HighIncome")
        .otherwise("MidAge_LowIncome")
    )

    with_segment = customer_features.withColumn("KPI_lifecycle_segment", segment)

    segments = [
        "Young_LowIncome",
        "MidAge_HighIncome",
        "Senior_Stable",
        "Young_HighIncome",
        "MidAge_LowIncome",
    ]
    flagged = with_segment
    for seg in segments:
        flagged = flagged.withColumn(
            f"FLAG_SEG_{seg}", (F.col("KPI_lifecycle_segment") == seg).cast("int")
        )
    return flagged.drop("KPI_lifecycle_segment")


def _action_quadrant(customer_features: DataFrame) -> DataFrame:
    """Business-rule heuristic segmenting customers into an action
    quadrant (Segment A/B/C/D), used ONLY to score the never-borrowed
    prospects for whom there is no real historical outcome to train a
    model against (see build_loan_level_features and
    pipelines/train_model.py for the fuller reasoning).

    This`assign_action_quadrant` rule -- deliberately NOT used as the ML target,
    since it's built out
    of several of the same columns (INCOME, volatility, repayment rate)
    that would otherwise be fed back in as predictors, which would make
    a model trained on it purely circular.
    """

    # -----------------------------------------------------------------
    # 1. OUTLIER MITIGATION LAYER (Winsorization via Cluster Quantiles)
    # -----------------------------------------------------------------
    # Compute the 1st, 50th (median), and 99th percentiles in a single cluster pass
    percentiles = customer_features.approxQuantile(
        ["INCOME", "KPI_net_cash_flow"], 
        [0.01, 0.5, 0.99], 
        0.01
    )
    
    # -----------------------------------------------------------------
    # BUSINESS STRATEGY HEURISTIC LAYER
    # -----------------------------------------------------------------
    # Extract income values
    income_median = percentiles[0][1] if percentiles and percentiles[0] else 0.0
    
    # Extract net cash flow boundaries (0=1st pct, 2=99th pct)
    cash_flow_p1 = percentiles[1][0] if percentiles and percentiles[1] else -2000.0
    cash_flow_p99 = percentiles[1][2] if percentiles and percentiles[1] else 5000.0

    # Apply the PySpark outlier cap using F.when logic to ground the tail leverage
    customer_features = customer_features.withColumn(
        "KPI_net_cash_flow",
        F.when(F.col("KPI_net_cash_flow") < cash_flow_p1, cash_flow_p1)
         .when(F.col("KPI_net_cash_flow") > cash_flow_p99, cash_flow_p99)
         .otherwise(F.col("KPI_net_cash_flow"))
    )


    income_median_row = customer_features.approxQuantile("INCOME", [0.5], 0.01)
    income_median = income_median_row[0] if income_median_row else 0.0

    is_untapped = (F.col("total_loans_taken") == 0) & (F.col("INCOME") >= income_median)
    is_high_risk = (
        (F.col("KPI_balance_volatility_index") > 1.2)
        | (F.col("KPI_low_balance_frequency") > 5)
        | ((F.col("total_loans_taken") > 0) & (F.col("KPI_repayment_track_record") < 0.70))
    )
    is_ideal = (
        (F.col("INCOME") >= income_median)
        & (F.col("KPI_balance_volatility_index") <= 1.2)
        & (F.col("KPI_net_cash_flow") > 0)
        & ((F.col("total_loans_taken") == 0) | (F.col("KPI_repayment_track_record") >= 0.90))
    )

    quadrant = (
        F.when(is_untapped, "Segment_C_Untapped_Potential")
        .when(is_high_risk, "Segment_B_High_Risk")
        .when(is_ideal, "Segment_A_Ideal_Target")
        .otherwise("Segment_D_Seasonal_Borrower")
    )
    return customer_features.withColumn("KPI_action_quadrant", quadrant)


def build_customer_features(
    customers: DataFrame,
    loans: DataFrame,
    transactions: DataFrame,
    thresholds: Dict[str, float],
) -> DataFrame:
    """Build the one-row-per-customer feature table.

    `thresholds` is expected to be the dict returned by
    `eda_checks.derive_empirical_thresholds` (keys:
    `low_balance_threshold`, `age_tertile_33`, `age_tertile_66`) -- passing
    it in explicitly (rather than recomputing thresholds inside this
    function) keeps the "what counts as low balance / what age splits the
    population" decision visible and auditable at the call site, and lets
    the same thresholds be reused consistently for both training and
    inference-time feature building.
    """
    borrower_ids = loans.select("ID").distinct().withColumn("is_borrower", F.lit(1))

    base = (
        customers.join(_transaction_aggregates(transactions), on="ID", how="left")
        .join(_loan_aggregates(loans), on="ID", how="left")
        .join(borrower_ids, on="ID", how="left")
    )

    base = base.withColumn(
        "customer_segment",
        F.when(F.col("is_borrower") == 1, "Existing Borrower").otherwise("Deposit-Only Prospect"),
    ).drop("is_borrower")

    # Customers who never appear in the loans table get zeros, not nulls,
    # for loan-derived counts -- "never borrowed" is a real, meaningful
    # value here, not missing data.
    for col in ["total_loans_taken", "successful_loans"]:
        base = base.withColumn(col, F.coalesce(F.col(col), F.lit(0)))
    for col in ["avg_balance", "balance_std", "total_transactions_count"]:
        base = base.withColumn(col, F.coalesce(F.col(col), F.lit(0.0)))

    base = base.join(_net_cash_flow(transactions), on="ID", how="left").withColumn(
        "KPI_net_cash_flow", F.coalesce(F.col("KPI_net_cash_flow"), F.lit(0.0))
    )

    base = base.withColumn(
        "KPI_balance_to_income_ratio",
        F.col("avg_balance") / F.when(F.col("INCOME") == 0, 1.0).otherwise(F.col("INCOME")),
    ).withColumn(
        "KPI_balance_volatility_index",
        F.col("balance_std")
        / F.when(F.col("avg_balance") == 0, 1.0).otherwise(F.col("avg_balance")),
    )

    base = base.join(
        _low_balance_frequency(transactions, thresholds["low_balance_threshold"]),
        on="ID",
        how="left",
    ).withColumn(
        "KPI_low_balance_frequency", F.coalesce(F.col("KPI_low_balance_frequency"), F.lit(0))
    )

    base = base.join(_burn_rate(transactions), on="ID", how="left").withColumn(
        "KPI_burn_rate_per_day", F.coalesce(F.col("KPI_burn_rate_per_day"), F.lit(0.0))
    )

    # KPI_loan_success_rate: historical take-up rate. -1.0 marks
    # "never applied" as a distinct category rather than an ambiguous 0,
    # since "never asked" and "always declined" are very different risk
    # profiles that a plain 0 would conflate.
    base = base.withColumn(
        "KPI_repayment_track_record",
        F.when(
            F.col("total_loans_taken") > 0,
            F.col("successful_loans") / F.col("total_loans_taken"),
        ).otherwise(F.lit(-1.0)),
    )

    base = base.withColumn(
        "KPI_loan_recency_days",
        F.coalesce(F.datediff(F.current_date(), F.col("last_loan_date")), F.lit(9999)),
    )

    base = base.withColumn(
        "KPI_max_proven_capacity", F.coalesce(F.col("max_loan_amount"), F.lit(0.0))
    ).withColumn(
        "KPI_credit_capacity_to_income",
        F.col("KPI_max_proven_capacity")
        / F.when(F.col("INCOME") == 0, 1.0).otherwise(F.col("INCOME")),
    )

    base = _lifecycle_segment_flags(
        base, thresholds["age_tertile_33"], thresholds["age_tertile_66"]
    )
    base = _action_quadrant(base)

    return base


# ---------------------------------------------------------------------------
# Loan-application-level feature table (used for modelling)
# ---------------------------------------------------------------------------

def build_loan_level_features(loans: DataFrame, customer_features: DataFrame) -> DataFrame:
    """Build the point-in-time-correct, loan-application-level table that
    the propensity model is actually trained on.

    Why loan-application level, and not customer level:
    ----------------------------------------------------
    The only genuine historical outcome in this dataset is `loans.OUTCOME`
    (TakeUp/Declined) -- an actual past decision. There is no real
    "would this customer take up a loan" label at the customer level; a
    customer-level target built from a hand-written rule (like
    `KPI_action_quadrant`) would just be teaching a model to reconstruct
    arithmetic we already wrote. So the model is trained at the grain
    where a real label exists: one row per historical loan application.

    Why point-in-time features, computed here rather than reused from
    `customer_features`:
    ----------------------------------------------------
    `customer_features.total_loans_taken`, `successful_loans`, and
    `KPI_repayment_track_record` are aggregated across a customer's ENTIRE
    loan history -- including the very application whose outcome we're
    trying to predict. Feeding that straight in as a predictor would leak
    each row's own label into its own features (a customer's first-ever
    application, if successful, would already show a "repayment track
    record" of 1.0 -- which only exists because it succeeded). To avoid
    that, this function recomputes the loan-history features using only
    applications strictly *before* the one being scored, via window
    functions ordered by DATE:

    - prior_loans_taken: count of earlier applications
      (Window.rowsBetween(unboundedPreceding, -1))
    - prior_successful_loans: count of earlier TakeUps
    - prior_max_loan_amount: largest amount successfully taken up before
      this application
    - prior_repayment_track_record: prior success rate, with -1.0 marking
      a customer's first-ever application (no history yet)

    Known limitation (documented, not silently ignored): the transaction-
    derived features (avg_balance, KPI_net_cash_flow, volatility, etc.)
    that get joined in below are still full-history aggregates, not
    re-computed per application date. Doing that properly would mean
    re-aggregating transactions for every application's as-of date, which
    is a much larger engineering lift than this case study's scope
    justifies -- but it does mean reported model performance is likely a
    modest over-estimate of what a fully point-in-time system would
    achieve in production. Worth flagging explicitly if this model is
    ever put into production as-is.
    """
    order_by_date = Window.partitionBy("ID").orderBy("DATE")
    prior_rows = order_by_date.rowsBetween(Window.unboundedPreceding, -1)

    with_flags = loans.withColumn("is_takeup", (F.col("OUTCOME") == "TakeUp").cast("int"))

    with_history = (
        with_flags.withColumn("prior_loans_taken", F.count(F.lit(1)).over(prior_rows))
        .withColumn("prior_successful_loans", F.coalesce(F.sum("is_takeup").over(prior_rows), F.lit(0)))
        .withColumn(
            "prior_max_loan_amount",
            F.coalesce(
                F.max(F.when(F.col("is_takeup") == 1, F.col("AMOUNT"))).over(prior_rows),
                F.lit(0.0),
            ),
        )
    )

    loan_level = with_history.withColumn(
        "prior_repayment_track_record",
        F.when(
            F.col("prior_loans_taken") > 0,
            F.col("prior_successful_loans") / F.col("prior_loans_taken"),
        ).otherwise(F.lit(-1.0)),
    ).withColumn("TARGET_propensity", F.col("is_takeup"))

    # Static, customer-level features that do NOT already encode this
    # application's own outcome -- safe to join in as-is. `customer_segment`
    # is deliberately excluded: at this grain every row already belongs to
    # a customer who has applied (that's what makes it a "loan" row), so
    # the segment is constant here and carries no information.
    static_columns = [
        "ID",
        "GENDER",
        "AGE",
        "INCOME",
        "avg_balance",
        "balance_std",
        "total_transactions_count",
        "KPI_net_cash_flow",
        "KPI_balance_to_income_ratio",
        "KPI_balance_volatility_index",
        "KPI_low_balance_frequency",
        "KPI_burn_rate_per_day",
    ] + [c for c in customer_features.columns if c.startswith("FLAG_SEG_")]

    loan_level = loan_level.join(
        customer_features.select(*static_columns), on="ID", how="left"
    ).withColumnRenamed("AMOUNT", "requested_amount")

    return loan_level


def get_static_feature_columns(customer_features: DataFrame) -> List[str]:
    """Return the customer-level column names used as static predictors
    at loan-application grain -- kept as a function (not a hardcoded
    module-level list) so it always reflects whatever lifecycle segment
    flags actually exist on the current `customer_features` table."""
    return [
        "GENDER",
        "AGE",
        "INCOME",
        "avg_balance",
        "balance_std",
        "total_transactions_count",
        "KPI_net_cash_flow",
        "KPI_balance_to_income_ratio",
        "KPI_balance_volatility_index",
        "KPI_low_balance_frequency",
        "KPI_burn_rate_per_day",
    ] + [c for c in customer_features.columns if c.startswith("FLAG_SEG_")]


def cap_outlier_kpis(
    df: DataFrame,
    kpi_cols: List[str] = ("KPI_net_cash_flow", "KPI_low_balance_frequency"),
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> DataFrame:
    """Winsorize the given KPI columns at the lower_q/upper_q percentiles.

    Same idea as clipping in pandas (`.clip(lower=..., upper=...)`), just
    computed via `approxQuantile` -- Spark's distributed quantile
    estimator -- instead of collecting the column. Overwrites each KPI
    in place rather than adding a "_clean" suffix, since every downstream
    consumer (modelling, plots) already refers to the KPI by its plain
    name; a suffixed column would need every one of those call sites
    updated too.
    """
    for col in kpi_cols:
        lower, upper = df.stat.approxQuantile(col, [lower_q, upper_q], 0.01)
        df = df.withColumn(
            col, F.when(F.col(col) < lower, lower).when(F.col(col) > upper, upper).otherwise(F.col(col))
        )
    return df