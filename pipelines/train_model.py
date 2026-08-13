"""
End-to-end training pipeline entry point.

Orchestrates the full flow: load raw data -> clean -> run EDA/data-quality
checks -> derive empirical thresholds -> build customer- and loan-level
features -> train/test split -> train the propensity model -> evaluate
(including the bias/fairness assessment) -> score never-borrowed prospects
-> persist the model, metrics, EDA report, and fairness report to disk.

Run with:

    python pipelines/train_model.py --config configs/training.yaml
"""

from __future__ import annotations

import argparse
import sys
import os
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use("Agg")  # headless: this script never shows a plot window, only saves files

from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from src import eda_checks as eda
from src import feature_selection as fsel
from src import features as feat
from src import modeling as model_lib
from src import viz
from src.data_prep import load_and_clean
from src.utils import ensure_parent_dir, get_spark_session, load_yaml, write_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the loan propensity model.")
    parser.add_argument("--config", type=str, required=True, help="Path to a training YAML config.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    output_cfg = cfg["output"]
    spark_cfg = cfg.get("spark", {})
    split_cfg = cfg.get("split", {})
    fsel_cfg = cfg.get("feature_selection", {})
    model_cfg = cfg.get("model", {})

    spark = get_spark_session(
        app_name=spark_cfg.get("app_name", "loan_propensity_training"),
        shuffle_partitions=spark_cfg.get("shuffle_partitions", 8),
    )

    # ---------------------------------------------------------------
    # 1. Load + clean
    # ---------------------------------------------------------------
    print("[1/7] Loading and cleaning raw data...")
    customers, loans, transactions = load_and_clean(
        spark,
        data_cfg["customers_path"],
        data_cfg["loans_path"],
        data_cfg["transactions_path"],
    )
    customers.cache()
    loans.cache()
    transactions.cache()

    # ---------------------------------------------------------------
    # 2. Empirically derive thresholds used by feature engineering
    #    (replaces hardcoded constants like "low balance = 500")
    # ---------------------------------------------------------------
    print("[2/7] Deriving empirical thresholds...")
    thresholds = eda.derive_empirical_thresholds(transactions, customers)
    print(f"       {thresholds}")

    # ---------------------------------------------------------------
    # 3. Build customer-level features (used for EDA, segmentation,
    #    and to score never-borrowed prospects at the end)
    # ---------------------------------------------------------------
    print("[3/7] Building customer-level features...")
    customer_features = feat.build_customer_features(customers, loans, transactions, thresholds)
    customer_features.cache()
    customer_features.count()  # materialise the cache before reuse below

    # ---------------------------------------------------------------
    # 3b. Time-series view: whole-portfolio burn rate by month, and the
    #    top individual customers by burn rate. This was in the original
    #    pandas prototype but got dropped when the notebook was first
    #    ported to PySpark -- reinstated here as a saved chart, since this
    #    script has no interactive display to show it in directly.
    # ---------------------------------------------------------------
    monthly_burn_trend = feat.compute_monthly_burn_rate_trend(transactions).toPandas()
    ax = viz.plot_monthly_trend(
        monthly_burn_trend, "year_month", "avg_burn_rate_per_day",
        "Portfolio Avg Burn Rate by Month",
    )
    ensure_parent_dir(output_cfg["burn_rate_trend_plot_path"])
    ax.figure.savefig(output_cfg["burn_rate_trend_plot_path"], bbox_inches="tight")

    top_burners = (
        customer_features.orderBy(F.desc("KPI_burn_rate_per_day"))
        .select("ID", "KPI_burn_rate_per_day")
        .limit(15)
        .toPandas()
    )
    ax = viz.plot_top_n_bar(
        top_burners, "ID", "KPI_burn_rate_per_day", "Top 15 Customers by Burn Rate"
    )
    ensure_parent_dir(output_cfg["top_burners_plot_path"])
    ax.figure.savefig(output_cfg["top_burners_plot_path"], bbox_inches="tight")
    print(f"       Time-series charts written to {output_cfg['burn_rate_trend_plot_path']} and {output_cfg['top_burners_plot_path']}")

    # ---------------------------------------------------------------
    # 4. Run the full EDA / data-quality report and persist it.
    #    This is run regardless of what the model finds -- data quality
    #    findings matter even if they don't change the model's headline
    #    metric, and silently skipping this if training "looks fine"
    #    would defeat the point of having it.
    # ---------------------------------------------------------------
    print("[4/7] Running EDA / data-quality checks...")
    eda_report = eda.run_full_eda_report(customers, loans, transactions, customer_features)
    write_yaml(output_cfg["eda_report_path"], eda_report)
    print(f"       EDA report written to {output_cfg['eda_report_path']}")
    if not eda_report["referential_integrity"]["passed"]:
        print("       WARNING: referential integrity check failed -- see EDA report for details.")

    # ---------------------------------------------------------------
    # 5. Build the loan-application-level table (point-in-time-correct
    #    features) and train/test split. This is the real supervised
    #    target: loans.OUTCOME, not the KPI_action_quadrant heuristic.
    # ---------------------------------------------------------------
    print("[5/7] Building loan-level features and splitting train/test...")
    loan_level = feat.build_loan_level_features(loans, customer_features)
    loan_level = model_lib.encode_gender(loan_level)

    static_cols = feat.get_static_feature_columns(customer_features)
    feature_columns = (
        [c for c in static_cols if c != "GENDER"]
        + ["GENDER_encoded", "GENDER_unknown"]
        + [
            "prior_loans_taken",
            "prior_successful_loans",
            "prior_max_loan_amount",
            "prior_repayment_track_record",
            "requested_amount",
        ]
    )

    train_df, test_df = model_lib.train_test_split(
        loan_level, test_fraction=split_cfg.get("test_fraction", 0.2), seed=split_cfg.get("seed", 42)
    )
    train_df.cache()
    test_df.cache()
    print(f"       train rows: {train_df.count()}, test rows: {test_df.count()}")

    # ---------------------------------------------------------------
    # 5b. Feature selection: drop low-variance features, prune
    #    multicollinear features by VIF, then rank what's left by ANOVA
    #    F-test discriminant power against TARGET_propensity.
    #
    #    Run strictly on train_df -- not loan_level -- so the test set
    #    has no influence on which features are even allowed into the
    #    model. Computing this on the full loan_level table (before the
    #    split) would leak test-set information into a decision that's
    #    supposed to happen before the model ever sees it, the same
    #    category of leak this project has been careful to close
    #    elsewhere (TARGET_propensity, point-in-time loan-history
    #    features).
    #
    #    Unlike a notebook (where a person can eyeball a VIF table and
    #    decide what to drop), this runs unattended, so VIF pruning is
    #    automated -- iteratively drop the single worst-VIF feature until
    #    everything remaining is at or below max_vif. See
    #    src/feature_selection.py for the full reasoning, including a
    #    documented caveat about one-hot feature families (FLAG_SEG_*)
    #    understating their own collinearity under this method.
    # ---------------------------------------------------------------
    print("       Running feature selection on train_df only (variance -> VIF -> ANOVA)...")
    feature_columns, dropped_low_variance, feature_variances = fsel.drop_low_variance_features(
        train_df, feature_columns, threshold=fsel_cfg.get("variance_threshold", 0.01)
    )
    if dropped_low_variance:
        print(f"       Dropped (low variance): {dropped_low_variance}")
    ax = viz.plot_variance_scores(feature_variances, threshold=fsel_cfg.get("variance_threshold", 0.01))
    ensure_parent_dir(output_cfg["variance_plot_path"])
    ax.figure.savefig(output_cfg["variance_plot_path"], bbox_inches="tight")

    vif_scores = fsel.calculate_vif(train_df, feature_columns)
    ax = viz.plot_vif_scores(vif_scores, max_vif=fsel_cfg.get("max_vif", 10.0))
    ensure_parent_dir(output_cfg["vif_plot_path"])
    ax.figure.savefig(output_cfg["vif_plot_path"], bbox_inches="tight")

    feature_columns, dropped_vif = fsel.select_features_by_vif(
        train_df, feature_columns, max_vif=fsel_cfg.get("max_vif", 10.0)
    )
    if dropped_vif:
        print(f"       Dropped (VIF pruning): {dropped_vif}")

    discriminant_scores = fsel.rank_features_anova(train_df, feature_columns, "TARGET_propensity")
    print("       ANOVA F-test ranking (top 5):")
    print(discriminant_scores.head(5).to_string(index=False))

    fig_ax = viz.plot_feature_selection_scores(discriminant_scores)
    ensure_parent_dir(output_cfg["feature_selection_plot_path"])
    fig_ax.figure.savefig(output_cfg["feature_selection_plot_path"], bbox_inches="tight")
    print("       Feature selection charts written to reports/ (variance, VIF, ANOVA)")

    # Explicit checkpoint: this is the exact feature list about to go into
    # training -- logged on purpose so feature selection's effect on
    # training is never just implied by an earlier step having run.
    print(f"       Training with {len(feature_columns)} features after selection:")
    for c in feature_columns:
        print(f"         - {c}")

    # ---------------------------------------------------------------
    # 6. Train, evaluate, and run the bias/fairness assessment
    # ---------------------------------------------------------------
    print("[6/7] Training model and running fairness assessment...")
    #pipeline_model, predictions, auc = model_lib.train_propensity_model(
    #    train_df, test_df, feature_columns
    #)

    pipeline_model, predictions, auc = model_lib.train_propensity_model(
    train_df, test_df, feature_columns,
    reg_param=model_cfg.get("reg_param", 0.1),
    elastic_net_param=model_cfg.get("elastic_net_param", 0.0),
)
    fairness = model_lib.fairness_report(predictions)
    #gender_ablation = model_lib.assess_gender_contribution(train_df, test_df, feature_columns)

    gender_ablation = model_lib.assess_gender_contribution(
    train_df, test_df, feature_columns,
    reg_param=model_cfg.get("reg_param", 0.1),
    elastic_net_param=model_cfg.get("elastic_net_param", 0.0),
)

    metrics = {
        "roc_auc": auc,
        "n_train": train_df.count(),
        "n_test": test_df.count(),
        "feature_columns": feature_columns,
        "fairness": fairness,
        "gender_ablation": gender_ablation,
    }
    write_yaml(output_cfg["metrics_path"], metrics)
    print(f"       ROC-AUC: {auc:.4f}")
    print(f"       Metrics written to {output_cfg['metrics_path']}")
    if fairness["four_fifths_rule_flag"]:
        print(
            "       WARNING: disparate impact ratio by gender is below the "
            "four-fifths benchmark -- see fairness section of metrics report."
        )

    # ---------------------------------------------------------------
    # 7. Score never-borrowed prospects. No ground truth exists for this
    #    group (they've never been offered a loan), so both the model's
    #    extrapolated score AND the Step-5 business-rule quadrant are
    #    reported side by side rather than presenting either alone as
    #    a validated answer -- see features.py's KPI_action_quadrant
    #    docstring for the reasoning.
    # ---------------------------------------------------------------
    print("[7/7] Scoring never-borrowed prospects...")
    prospects = customer_features.filter("total_loans_taken = 0")
    prospects = model_lib.encode_gender(prospects)

    # Prospects have no loan-history row to compute prior_* features from;
    # use the "no history yet" conventions established in features.py.
    prospects_for_scoring = (
        prospects.withColumn("prior_loans_taken", F.lit(0))
        .withColumn("prior_successful_loans", F.lit(0))
        .withColumn("prior_max_loan_amount", F.lit(0.0))
        .withColumn("prior_repayment_track_record", F.lit(-1.0))
        .withColumn("requested_amount", F.lit(0.0))
    )
    # `probability` is a Spark ML Vector (dense/sparse struct), not a plain
    # array -- `vector_to_array` converts it so a normal column index works.
    scored = pipeline_model.transform(prospects_for_scoring).withColumn(
        "probability_array", vector_to_array(F.col("probability"))
    )
    prospect_scores = scored.select(
        "ID",
        "customer_segment",
        "KPI_action_quadrant",
        F.col("probability_array").getItem(1).alias("model_propensity_score"),
    )

    model_path = output_cfg["model_path"]
    pipeline_model.write().overwrite().save(model_path)
    prospect_scores.write.mode("overwrite").parquet(output_cfg["prospect_scores_path"])
    print(f"       Model saved to {model_path}")
    print(f"       Prospect scores saved to {output_cfg['prospect_scores_path']}")

    spark.stop()


if __name__ == "__main__":
    main()
