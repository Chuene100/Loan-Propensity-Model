"""
Model training and evaluation, in PySpark ML.

Trains a logistic regression propensity model on the loan-application-
level table from `features.build_loan_level_features` (target =
`TARGET_propensity`, i.e. did loans.OUTCOME == 'TakeUp'), then runs the
bias/fairness assessment the case study explicitly asks for
("is there bias/discrimination in the data") -- this was entirely absent
from earlier drafts of this analysis, so it's treated here as a first-
class part of modelling, not an optional add-on.

Two things this module deliberately does NOT do:
- It does not use `KPI_action_quadrant` as a training target (see
  features.build_loan_level_features's docstring for why that would be
  circular).
- It does not silently include GENDER without checking whether it's
  earning its place -- see `assess_gender_contribution` below.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def encode_gender(loan_level: DataFrame) -> DataFrame:
    """Encode GENDER as a numeric column, with an explicit "unknown" flag.

    customers.GENDER was already normalised to 'Female'/'Male'/'Unknown'
    in data_prep.clean_customers. Here we map 'Female' -> 0, 'Male' -> 1,
    and anything else (i.e. 'Unknown') -> -1, with a separate
    GENDER_unknown indicator column. Keeping "unknown" as its own visible
    category -- rather than folding it into one of the two known
    genders by default -- avoids quietly misrepresenting missing data as
    a real value, which is exactly the bug found in the original
    notebook's gender-mapping cell.
    """
    return (
        loan_level.withColumn(
            "GENDER_unknown", (~F.col("GENDER").isin("Female", "Male")).cast("int")
        )
        .withColumn(
            "GENDER_encoded",
            F.when(F.col("GENDER") == "Female", 0)
            .when(F.col("GENDER") == "Male", 1)
            .otherwise(-1),
        )
    )


def build_pipeline(
    feature_columns: List[str],
    label_column: str = "TARGET_propensity",
    reg_param: float = 0.1,
    elastic_net_param: float = 0.0,  # 0.0 = pure L2 (Ridge); 1.0 = pure L1 (Lasso)
) -> Pipeline:
    """Build a Spark ML Pipeline: assemble features -> scale -> logistic
    regression.

    `class_weight='balanced'` doesn't exist in Spark ML's
    LogisticRegression; the Spark-native equivalent is a per-row weight
    column, computed here as the inverse class frequency and passed via
    `weightCol`. This achieves the same "don't let the majority class
    dominate the loss" effect as scikit-learn's class_weight parameter,
    just expressed as data (a column) instead of a model hyperparameter,
    which is how Spark ML expects it.
    """
    assembler = VectorAssembler(
        inputCols=feature_columns, outputCol="features_raw", handleInvalid="keep"
    )
    scaler = StandardScaler(
        inputCol="features_raw", outputCol="features", withMean=True, withStd=True
    )
    classifier = LogisticRegression(
        featuresCol="features",
        labelCol=label_column,
        weightCol="class_weight",
        maxIter=100,
        regParam=reg_param,
        elasticNetParam=elastic_net_param,
    )
    return Pipeline(stages=[assembler, scaler, classifier])


def add_class_weight_column(
    df: DataFrame, label_column: str = "TARGET_propensity"
) -> DataFrame:
    """Add a `class_weight` column giving each row weight inversely
    proportional to its class's frequency, so the minority class isn't
    swamped by the majority class during training -- the Spark ML
    equivalent of scikit-learn's `class_weight='balanced'`.
    """
    counts = df.groupBy(label_column).count().collect()
    total = sum(row["count"] for row in counts)
    n_classes = len(counts)
    weight_map = {row[label_column]: total / (n_classes * row["count"]) for row in counts}

    weight_expr = F.lit(1.0)
    for label_value, weight in weight_map.items():
        weight_expr = F.when(F.col(label_column) == label_value, F.lit(weight)).otherwise(
            weight_expr
        )
    return df.withColumn("class_weight", weight_expr)


def train_test_split(
    df: DataFrame, label_column: str = "TARGET_propensity", test_fraction: float = 0.15, seed: int = 42
) -> Tuple[DataFrame, DataFrame]:
    """Stratified-ish train/test split.

    Spark's `randomSplit` alone doesn't guarantee the same class ratio in
    both halves for a small/imbalanced dataset the way sklearn's
    `stratify=` does. To approximate stratification, split each class
    independently with the same fraction and union the results back
    together -- for the two classes in this problem (TakeUp/Declined)
    that's a cheap, exact way to preserve the class ratio in both splits.
    """
    train_parts, test_parts = [], []
    for label_value, in df.select(label_column).distinct().collect():
        subset = df.filter(F.col(label_column) == label_value)
        train_subset, test_subset = subset.randomSplit(
            [1 - test_fraction, test_fraction], seed=seed
        )
        train_parts.append(train_subset)
        test_parts.append(test_subset)

    train_df = train_parts[0]
    for part in train_parts[1:]:
        train_df = train_df.union(part)
    test_df = test_parts[0]
    for part in test_parts[1:]:
        test_df = test_df.union(part)
    return train_df, test_df


def evaluate_auc(
    predictions: DataFrame, label_column: str = "TARGET_propensity"
) -> float:
    """Compute ROC-AUC for a predictions DataFrame containing a
    `probability` column (Spark ML's standard output column name)."""
    evaluator = BinaryClassificationEvaluator(
        labelCol=label_column, rawPredictionCol="probability", metricName="areaUnderROC"
    )
    return evaluator.evaluate(predictions)


def train_propensity_model(
    train_df: DataFrame,
    test_df: DataFrame,
    feature_columns: List[str],
    label_column: str = "TARGET_propensity",
    reg_param: float = 0.1,
    elastic_net_param: float = 0.0,
) -> Tuple[PipelineModel, DataFrame, float]:
    """Fit the pipeline on `train_df`, score `test_df`, and return the
    fitted model, the test-set predictions, and the test-set AUC."""
    train_weighted = add_class_weight_column(train_df, label_column)
    pipeline = build_pipeline(feature_columns, label_column, reg_param, elastic_net_param)
    model = pipeline.fit(train_weighted)
    predictions = model.transform(test_df)
    auc = evaluate_auc(predictions, label_column)
    return model, predictions, auc


def assess_gender_contribution(
    train_df: DataFrame,
    test_df: DataFrame,
    feature_columns: List[str],
    label_column: str = "TARGET_propensity",
    reg_param: float = 0.1,
    elastic_net_param: float = 0.0,
) -> Dict:
    """Compare model AUC with vs. without GENDER-derived columns, to check
    whether a protected attribute is actually earning its place as a
    predictor, or just riding along at a fairness cost with no real lift.

    If dropping GENDER/GENDER_unknown barely moves AUC, there's little
    predictive reason to keep a protected characteristic in a live
    lending model -- removing it is generally the more defensible default
    unless there's a specific, articulated reason to keep it.
    """
    gender_cols = [c for c in feature_columns if c.startswith("GENDER")]
    reduced_columns = [c for c in feature_columns if c not in gender_cols]

    _, _, auc_with = train_propensity_model(train_df, test_df, feature_columns, label_column, reg_param, elastic_net_param)
    _, _, auc_without = train_propensity_model(train_df, test_df, reduced_columns, label_column, reg_param, elastic_net_param)

    return {
        "auc_with_gender": auc_with,
        "auc_without_gender": auc_without,
        "auc_difference": auc_with - auc_without,
        "interpretation": (
            "Gender contributes meaningfully to discrimination power; its removal "
            "should be weighed against this predictive cost."
            if abs(auc_with - auc_without) > 0.01
            else "Gender adds little to no measurable predictive power here; "
            "excluding it from the live model costs little and removes a "
            "protected characteristic from a lending decision."
        ),
    }


def fairness_report(
    predictions: DataFrame, label_column: str = "TARGET_propensity"
) -> Dict:
    """Group-wise fairness assessment on the model's predictions.

    Reports the predicted-positive ("would offer a loan") rate by GENDER
    and by AGE band, plus the disparate impact ratio (min group rate /
    max group rate) across gender groups. This directly answers the case
    study's explicit question -- "is there bias/discrimination in the
    data" -- rather than leaving it implicit or unaddressed.

    The commonly cited "four-fifths rule" (a disparate impact ratio below
    0.80 warranting scrutiny) is a US EEOC convention, used here only as a
    rough, widely-recognised benchmark for flagging a ratio worth a closer
    look -- not as a legal threshold in any particular jurisdiction.
    """
    with_prediction = predictions.withColumn(
        "predicted_positive", F.col("prediction").cast("int")
    )

    by_gender = (
        with_prediction.groupBy("GENDER")
        .agg(F.avg("predicted_positive").alias("predicted_approval_rate"))
        .toPandas()
    )
    rates = by_gender["predicted_approval_rate"]
    disparate_impact_ratio = float(rates.min() / rates.max()) if len(rates) > 1 else 1.0

    age_banded = with_prediction.withColumn(
        "age_band",
        F.when(F.col("AGE") <= 25, "<=25")
        .when(F.col("AGE") <= 35, "26-35")
        .when(F.col("AGE") <= 50, "36-50")
        .when(F.col("AGE") <= 65, "51-65")
        .otherwise("65+"),
    )
    by_age = (
        age_banded.groupBy("age_band")
        .agg(F.avg("predicted_positive").alias("predicted_approval_rate"))
        .toPandas()
    )

    return {
        "approval_rate_by_gender": by_gender.set_index("GENDER")[
            "predicted_approval_rate"
        ].to_dict(),
        "approval_rate_by_age_band": by_age.set_index("age_band")[
            "predicted_approval_rate"
        ].to_dict(),
        "disparate_impact_ratio_gender": disparate_impact_ratio,
        "four_fifths_rule_flag": disparate_impact_ratio < 0.80,
    }

def compute_confusion_matrix(predictions: DataFrame, label_column: str = "TARGET_propensity") -> Dict[str, int]:
    """Confusion matrix counts (TN, FP, FN, TP), computed as a single
    distributed groupBy over (label, prediction) -- only four numbers are
    ever collected to the driver, same discipline as the fairness report
    above.
    """
    counts = predictions.groupBy(label_column, "prediction").count().toPandas()

    def get(actual: int, pred: float) -> int:
        row = counts[(counts[label_column] == actual) & (counts["prediction"] == pred)]
        return int(row["count"].iloc[0]) if len(row) else 0

    return {
        "true_negative": get(0, 0.0),
        "false_positive": get(0, 1.0),
        "false_negative": get(1, 0.0),
        "true_positive": get(1, 1.0),
    }


def find_threshold_for_target_rate(predictions: DataFrame, target_rate: float = 0.143) -> float:
    """Starting-point threshold: the cutoff whose predicted-positive rate
    matches the historical base rate. Treat this as a sane default to
    move from, not the final answer -- the real threshold should come
    from the cost trade-off between a wasted promo (false positive) and
    a missed conversion (false negative), same discussion as the
    confusion-matrix slide."""
    from pyspark.ml.functions import vector_to_array
    pdf = (
        predictions.withColumn("prob_array", vector_to_array(F.col("probability")))
        .select(F.col("prob_array").getItem(1).alias("predicted_prob"))
        .toPandas()
    )
    return float(pdf["predicted_prob"].quantile(1 - target_rate))

def compare_approval_rates(
    predictions: DataFrame, historical_base_rate: float, threshold: float
) -> Dict[str, float]:
    """
    Three numbers, side by side: the historical base rate, the model's
    predicted-positive rate at Spark ML's default 0.5 cutoff, and the
    predicted-positive rate at the rebased threshold from
    find_threshold_for_target_rate. This is the direct visual answer to
    "why doesn't the model's approval rate match the historical rate" --
    class-weighted training shifts the default-threshold number away from
    the base rate on purpose; the rebased number should land close to it
    by construction.
    """
    from pyspark.ml.functions import vector_to_array

    with_prob = predictions.withColumn(
        "predicted_prob", vector_to_array(F.col("probability")).getItem(1)
    )
    rate_default = with_prob.filter(F.col("predicted_prob") >= 0.5).count() / with_prob.count()
    rate_rebased = with_prob.filter(F.col("predicted_prob") >= threshold).count() / with_prob.count()

    return {
        "historical_base_rate": historical_base_rate,
        "predicted_rate_default_threshold": rate_default,
        "predicted_rate_rebased_threshold": rate_rebased,
        "threshold_used": threshold,
    }

