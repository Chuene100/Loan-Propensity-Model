"""
Data loading and cleaning, in PySpark.

This module owns Step 0/1 of the case study: read the three raw tables
and fix the data-quality issues found during EDA, before any feature
engineering happens. Every cleaning decision here mirrors a specific
finding from the exploratory notebook (see notebooks/ and the docstrings
below for the "why"), rather than being an unexplained default.

Design note on file formats: the original case-study data ships as
Parquet, but the project skeleton's config used CSV paths. `load_data`
accepts either -- it dispatches on the file extension -- so the same
code works whether you're pointed at the original .parquet files or at
CSV exports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

CUSTOMERS_SCHEMA = StructType(
    [
        StructField("ID", StringType(), nullable=False),
        StructField("GENDER", StringType(), nullable=True),
        StructField("AGE", DoubleType(), nullable=True),
        StructField("INCOME", DoubleType(), nullable=True),
    ]
)

LOANS_SCHEMA = StructType(
    [
        StructField("ID", StringType(), nullable=False),
        StructField("DATE", StringType(), nullable=True),  # parsed to date below
        StructField("AMOUNT", DoubleType(), nullable=True),
        StructField("OUTCOME", StringType(), nullable=True),
    ]
)

TRANSACTIONS_SCHEMA = StructType(
    [
        StructField("ID", StringType(), nullable=False),
        StructField("DATE", StringType(), nullable=True),  # parsed to date below
        StructField("AMOUNT", DoubleType(), nullable=True),
        StructField("BALANCE", DoubleType(), nullable=True),
    ]
)


def _resolve_input_path(path: str) -> str:
    """Resolve an input path from the current working directory or the project root."""
    raw_path = Path(path)
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [raw_path]

    if not raw_path.is_absolute():
        candidates.extend([
            Path.cwd() / raw_path,
            repo_root / raw_path,
        ])

        # Support notebook-style paths such as ../data/customers.parquet.
        stripped_parts = [part for part in raw_path.parts if part not in {"", "."}]
        while stripped_parts and stripped_parts[0] == "..":
            stripped_parts.pop(0)
        if stripped_parts:
            normalized_path = Path(*stripped_parts)
            candidates.extend([repo_root / normalized_path, Path.cwd() / normalized_path])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    if raw_path.suffix.lower() == ".csv":
        parquet_candidate = raw_path.with_suffix(".parquet")
        repo_root_candidate = repo_root / parquet_candidate
        cwd_candidate = Path.cwd() / parquet_candidate
        for candidate in [parquet_candidate, repo_root_candidate, cwd_candidate]:
            if candidate.exists():
                return str(candidate)

    return str(raw_path)


def _read_any(spark: SparkSession, path: str, schema: StructType) -> DataFrame:
    """Read a CSV or Parquet file into a DataFrame with an explicit schema.

    An explicit schema (rather than `inferSchema`) is deliberate: schema
    inference on CSV requires a full extra pass over the data to guess
    types, and can silently guess wrong on edge cases (e.g. an all-numeric
    ID column being inferred as a long, then losing leading zeros).
    """
    resolved_path = _resolve_input_path(path)
    if resolved_path.endswith(".parquet"):
        # Parquet is self-describing, so we read it as-is and then cast
        # to the target schema explicitly (covers cases where the source
        # parquet stored, say, AGE as a long rather than a double).
        df = spark.read.parquet(resolved_path)
        for field in schema.fields:
            if field.name in df.columns:
                df = df.withColumn(field.name, F.col(field.name).cast(field.dataType))
        return df
    return spark.read.csv(resolved_path, header=True, schema=schema)


def load_data(
    spark: SparkSession, customers_path: str, loans_path: str, transactions_path: str
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Load the three raw tables as Spark DataFrames (uncleaned)."""
    customers = _read_any(spark, customers_path, CUSTOMERS_SCHEMA)
    loans = _read_any(spark, loans_path, LOANS_SCHEMA)
    transactions = _read_any(spark, transactions_path, TRANSACTIONS_SCHEMA)
    return customers, loans, transactions


def clean_customers(customers: DataFrame) -> DataFrame:
    """Clean the customers table.

    Fixes applied (each one a real issue found in EDA on this dataset):

    1. GENDER normalisation. Raw values include mixed case and
       whitespace (" male", "FEMALE", "F", "M", ...). We strip/upper
       first and *then* map to a canonical 'Female'/'Male', so a variant
       is normalised instead of silently falling through to a default
       value -- the original notebook had a bug here where the cleaning
       was only ever printed as a diagnostic, never applied, and a
       downstream `.fillna(0)` quietly counted every unrecognised
       category as Female. Anything left unrecognised after normalising
       is kept as 'Unknown' rather than guessed at.
    2. AGE outlier handling. `AGE` values above 120 are clearly invalid
       row-level errors (e.g. simulated sentinel values), not real ages.
       They're set to null and then imputed with the column median,
       computed with `approxQuantile` (Spark's distributed, single-pass
       quantile estimator -- exact quantiles require a full sort, which
       doesn't scale, so `approxQuantile`'s bounded relative error is the
       right trade-off here).
    3. INCOME imputation. Missing income is imputed with the median
       income. We deliberately do *not* touch extreme-but-plausible high
       incomes here -- that's a modelling decision (whether to cap
       outliers) that belongs with feature engineering, not cleaning.
    """
    cleaned = customers.withColumn(
        "GENDER",
        F.upper(F.trim(F.col("GENDER"))),
    ).withColumn(
        "GENDER",
        F.when(F.col("GENDER").isin("FEMALE", "F"), "Female")
        .when(F.col("GENDER").isin("MALE", "M"), "Male")
        .otherwise("Unknown"),
    )

    # AGE: treat > 120 as invalid, then median-impute.
    invalid_age = (F.col("AGE") > 120) | (F.col("AGE") < 0)
    cleaned = cleaned.withColumn("AGE", F.when(invalid_age, None).otherwise(F.col("AGE")))
    age_median = cleaned.stat.approxQuantile("AGE", [0.5], 0.01)[0]
    cleaned = cleaned.withColumn(
        "AGE", F.coalesce(F.col("AGE"), F.lit(age_median)).cast("int")
    )

    # INCOME: median-impute missing values only.
    income_median = cleaned.stat.approxQuantile("INCOME", [0.5], 0.01)[0]
    cleaned = cleaned.withColumn(
        "INCOME", F.coalesce(F.col("INCOME"), F.lit(income_median))
    )

    return cleaned


def clean_loans(loans: DataFrame) -> DataFrame:
    """Clean the loans (loan application) table.

    Parses DATE to a proper date type and ensures AMOUNT/OUTCOME have
    predictable types. We do NOT drop rows with a null OUTCOME here --
    that's a decision for the caller, since a null outcome likely means
    "application still pending" rather than bad data, and silently
    dropping it would bias the historical TakeUp/Declined base rate.
    """
    return loans.withColumn("DATE", F.to_date("DATE")).withColumn(
        "AMOUNT", F.col("AMOUNT").cast("double")
    )


def clean_transactions(transactions: DataFrame) -> DataFrame:
    """Clean the transactions table.

    - Parses DATE to a proper date type.
    - Flags rows with a missing BALANCE (`BALANCE_MISSING`) *before*
      imputing, so the flag itself can be inspected or used as a feature,
      rather than the imputation silently erasing the fact that a value
      was missing.
    - Imputes missing BALANCE with the column median (`approxQuantile`,
      same rationale as AGE above).
    - Negative BALANCE values are left as-is. Whether a negative balance
      represents a valid overdraft or a data artefact is an open EDA
      question (see eda_checks.negative_balance_report) -- cleaning
      should not silently decide that for you by clipping to zero.
    """
    cleaned = transactions.withColumn("DATE", F.to_date("DATE")).withColumn(
        "AMOUNT", F.col("AMOUNT").cast("double")
    )
    cleaned = cleaned.withColumn("BALANCE_MISSING", F.col("BALANCE").isNull().cast("int"))
    balance_median = cleaned.stat.approxQuantile("BALANCE", [0.5], 0.01)[0]
    cleaned = cleaned.withColumn(
        "BALANCE", F.coalesce(F.col("BALANCE"), F.lit(balance_median))
    )
    return cleaned


def load_and_clean(
    spark: SparkSession, customers_path: str, loans_path: str, transactions_path: str
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Convenience wrapper: load the three tables and apply all cleaning."""
    customers, loans, transactions = load_data(
        spark, customers_path, loans_path, transactions_path
    )
    return clean_customers(customers), clean_loans(loans), clean_transactions(transactions)
