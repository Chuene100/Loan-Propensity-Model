import os
import time
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

try:
    from src.modeling import NUMERIC_FEATURES, CATEGORICAL_FEATURES
except Exception:
    NUMERIC_FEATURES = []
    CATEGORICAL_FEATURES = []

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "models/loan_propensity_rf.joblib")
model = None

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None
except Exception as exc:
    raise RuntimeError(f"Unable to load model from {MODEL_PATH}: {exc}") from exc

PREDICTION_COUNTER = Counter(
    "loan_propensity_predictions_total",
    "Total predictions",
    ["status"],
)
PREDICTION_LATENCY = Histogram(
    "loan_propensity_prediction_latency_seconds",
    "Prediction latency",
)
SCORE_HISTOGRAM = Histogram(
    "loan_propensity_score",
    "Distribution of predicted propensity scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


class CustomerFeatures(BaseModel):
    ID: str
    AGE: int
    INCOME: float
    GENDER_ENC: int
    txn_count: float
    inflow_count: float
    outflow_count: float
    total_inflow: float
    total_outflow: float
    net_cashflow: float
    avg_balance: float
    min_balance: float
    std_balance: float
    balance_to_income_ratio: float
    txn_per_month: float
    total_loans: float
    num_declined: float
    max_loan_amount: float
    mean_loan_amount: float


@app.post("/score")
def score(payload: List[CustomerFeatures]):
    start = time.time()
    if model is None:
        PREDICTION_COUNTER.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Model is not available")

    if not NUMERIC_FEATURES or not CATEGORICAL_FEATURES:
        PREDICTION_COUNTER.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Model feature definitions are unavailable")

    try:
        df = pd.DataFrame([p.dict() for p in payload])
        X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        scores = model.predict_proba(X)[:, 1]
        for s in scores:
            SCORE_HISTOGRAM.observe(s)
        result = df[["ID"]].copy()
        result["propensity_score"] = scores
        PREDICTION_COUNTER.labels(status="ok").inc()
        PREDICTION_LATENCY.observe(time.time() - start)
        return result.to_dict(orient="records")
    except Exception as e:
        PREDICTION_COUNTER.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
