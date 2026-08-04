# Monitoring: Prometheus + Grafana

Wires the serving API's `/metrics` endpoint into Prometheus, and Prometheus
into a pre-provisioned Grafana dashboard -- both come up configured, no
manual "add data source" or "import dashboard" clicking required.

## Quick start

```bash
docker compose -f ../docker-compose.monitoring.yml up --build
```

Then open:
- **Grafana**: http://localhost:3000 (login `admin` / `admin` by default --
  override with a `GRAFANA_ADMIN_PASSWORD` environment variable before
  running `docker compose up`). The "Loan Propensity" folder has one
  dashboard, "Loan Propensity Service — Overview", already loaded.
- **Prometheus**: http://localhost:9090 -- useful for testing a PromQL
  expression directly before adding it to a panel.
- **Serving API**: http://localhost:8000/metrics -- the raw Prometheus
  exposition text Grafana is ultimately reading.

## What's on the dashboard

| Panel | What it answers |
|---|---|
| Service Up | Is Prometheus currently able to scrape the serving API at all? |
| Prediction Rate | How many `/score` calls per second, right now? |
| Error Rate (%) | What fraction of calls are failing? |
| p95 Prediction Latency | Is the service slow for the *typical* worst-case request? |
| Mean Propensity Score | Sanity-check number -- a sudden jump often means something upstream changed |
| Total Predictions (24h) | Volume, for capacity/cost conversations |
| Prediction Rate by Status | Same as above, split ok vs. error, over time |
| Prediction Latency Percentiles | p50/p95/p99 together -- catches a long tail a single average would hide |
| Propensity Score Distribution Over Time | Heatmap -- the fastest way to *see* a scoring-behaviour shift |
| Current Score Bucket Counts | Same distribution, last 15 minutes, as a bar chart |

## How this was validated (no Docker available in the build environment)

Every piece here was actually tested, not just written and assumed correct:

1. **A real bug was found and fixed in `serving/app.py`**: it imported
   `CONTENT_TYPE` from `prometheus_client`, which doesn't exist (the real
   name is `CONTENT_TYPE_LATEST`) -- as written, the app would have crashed
   at import time, so Prometheus would have had nothing to scrape. Fixed.
2. **The `/metrics` endpoint was actually run** (via a minimal harness
   replicating the exact metric definitions) and its exposition output
   confirmed to match every metric name referenced in the dashboard's
   PromQL queries.
3. **`monitoring/prometheus.yml` was validated with `promtool check
   config`.**
4. **A real Prometheus binary was run locally**, pointed at the harness
   above, and **every single PromQL query in the dashboard JSON was
   executed against it via the HTTP API** -- all 12 returned
   `"status": "success"` with non-empty results.

What wasn't tested end-to-end: the actual Docker Compose stack (no Docker
daemon in the build environment) and Grafana's own rendering of the
dashboard (no way to run the Grafana container itself). The provisioning
YAML follows Grafana's documented schema exactly, and the queries behind
every panel are independently confirmed correct against a real Prometheus
-- so the remaining risk is narrowly about Grafana/Compose plumbing, not
query correctness.

## Known gap this doesn't fix

`serving/app.py`'s `/score` endpoint still expects the earlier
scikit-learn feature set (see the main README's "Known gap" section) and
needs a real `models/loan_propensity_rf.joblib` to start at all. The
monitoring wiring in this folder is correct and tested independently of
that gap, but the container won't actually boot until that model file
exists or the endpoint is updated to match the current PySpark pipeline.
