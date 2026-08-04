# Loan Propensity MLOps Project

This project builds a customer-level loan-offer propensity model, in
**PySpark**, with basic MLOps scaffolding around it.

## What's here

- `src/data_prep.py` -- load the three raw tables (Parquet or CSV) and
  apply the data-quality fixes found during EDA (gender normalisation,
  age-outlier handling, income/balance imputation).
- `src/eda_checks.py` -- referential integrity, outlier/sentinel
  detection, distribution-shift tests (KS-test, chi-square), negative-
  balance auditing, transaction-history sufficiency, class balance, and
  data-driven threshold derivation. Also includes a Population Stability
  Index (PSI) function for monitoring feature drift over time.
- `src/features.py` -- customer-level financial-health KPIs (net cash
  flow, balance volatility, burn rate, credit capacity, lifecycle
  segments) and the point-in-time-correct loan-application-level feature
  table the model is actually trained on.
- `src/modeling.py` -- Spark ML training pipeline (VectorAssembler ->
  StandardScaler -> LogisticRegression), class-imbalance handling,
  stratified train/test split, and the bias/fairness assessment
  (disparate impact by gender/age, a with-vs-without-gender AUC
  comparison).
- `pipelines/train_model.py` -- orchestrates the full flow end-to-end and
  persists the model, EDA report, metrics (incl. fairness), and scored
  prospects to disk.
- `notebooks/loan_propensity_case_study.ipynb` -- the exploratory
  companion to the pipeline above: same functions, narrated with EDA
  findings and the reasoning behind each modelling decision. Kept thin
  deliberately -- the real logic lives in `src/`, reusable outside a
  notebook, so there is one source of truth.

## Why the target variable is `loans.OUTCOME`, not a business rule

The one genuine historical label in this dataset is `loans.OUTCOME`
(TakeUp/Declined) -- an actual past decision. Earlier iterations of this
analysis built a target out of a hand-written segmentation rule
(`KPI_action_quadrant`), then fed several of that same rule's inputs back
in as model features -- which is circular: the model just re-derives
arithmetic that was already written, and looks deceptively accurate doing
it. This version trains on `loans.OUTCOME` instead, at loan-application
grain, with loan-history features computed point-in-time (via Spark
window functions) so no application's own outcome leaks into its own
inputs. The rule-based quadrant is kept, but only to score the
never-borrowed prospects for whom no real outcome exists to validate a
model against -- see `features.build_loan_level_features`'s docstring for
the full reasoning.

## Quick start

1. Place your data in `data/customers.{csv,parquet}`,
   `data/loans.{csv,parquet}`, `data/transactions.{csv,parquet}` (either
   format works -- `src/data_prep.py` dispatches on the extension).
2. Install dependencies (PySpark needs a JVM at runtime -- see
   `docker/Dockerfile.training` if you don't have Java locally):

   ```bash
   pip install -r requirements.txt
   ```

3. Train a model:

   ```bash
   python pipelines/train_model.py --config configs/training.yaml
   ```

   This writes:
   - `models/loan_propensity_spark_pipeline/` -- the fitted Spark
     PipelineModel
   - `reports/eda_report.yaml` -- the full data-quality/EDA report
   - `reports/training_metrics.yaml` -- AUC, fairness report, and the
     gender-ablation comparison
   - `reports/prospect_scores.parquet` -- model + business-rule scores
     for never-borrowed prospects

4. Or explore interactively via
   `notebooks/loan_propensity_case_study.ipynb`.

5. Build and run the serving image (example):

   ```bash
   docker build -f docker/Dockerfile.serving -t loan-propensity-service .
   docker run -p 8000:8000 loan-propensity-service
   ```

6. Monitoring is already wired up -- Prometheus scrapes the serving API,
   and Grafana comes pre-provisioned with a dashboard (request rate, error
   rate, latency percentiles, and score-distribution drift):

   ```bash
   docker compose -f docker-compose.monitoring.yml up --build
   ```

   Grafana: http://localhost:3000 (`admin`/`admin` by default). See
   `monitoring/README.md` for the panel-by-panel breakdown and exactly how
   this was validated without a Docker daemon available in the build
   environment (a real Prometheus binary was run locally and every
   dashboard query executed against it).

## Known gap: `serving/app.py` is not yet updated for this pipeline

One real bug in `serving/app.py` *was* fixed as part of wiring up
monitoring: it imported `CONTENT_TYPE` from `prometheus_client`, which
doesn't exist (`CONTENT_TYPE_LATEST` is the correct name) -- as written,
the app would have crashed at import time and Prometheus would have had
nothing to scrape. That's fixed.

What's still open: `serving/app.py` still reflects the earlier
scikit-learn feature set
(`NUMERIC_FEATURES`/`CATEGORICAL_FEATURES` imported from a module that no
longer exists in that form) and has **not** been rewritten as part of this
change -- rewriting it properly is a separate decision, not an oversight:

Spark isn't a great fit for a low-latency, single-row REST endpoint the
way `serving/app.py` is built -- a Spark session has real JVM startup and
per-request overhead that a synchronous scikit-learn-style `/score`
endpoint doesn't. Two reasonable paths forward, worth deciding on
purpose rather than defaulting into one:

- **Batch scoring** (already implemented): run `pipelines/train_model.py`
  periodically and serve `reports/prospect_scores.parquet` from
  wherever downstream systems read it. This is usually the right choice
  for "should we approach this customer with a loan offer" -- a decision
  that doesn't need sub-second latency.
- **Export a lightweight model for real-time serving**: Spark ML's
  `LogisticRegressionModel` coefficients can be extracted and re-applied
  with a small amount of NumPy/pandas code (no Spark dependency) inside
  `serving/app.py`, if a true low-latency endpoint is actually needed.

Until one of those is implemented, treat `serving/` as stale relative to
the rest of this project.

Extend this baseline with your own orchestration, feature store, and
CI/CD.
