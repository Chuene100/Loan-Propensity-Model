"""
Generic helpers shared across the pipeline: Spark session construction,
YAML I/O, and small formatting utilities used when writing human-readable
reports (EDA findings, metrics, fairness summaries) to disk.

Keeping these in one place means every module (data_prep, features,
eda_checks, modeling) builds/reuses the *same* SparkSession rather than
each creating its own, which would spin up redundant JVM contexts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml
from pyspark.sql import SparkSession


def get_spark_session(app_name: str = "loan_propensity", shuffle_partitions: int = 8) -> SparkSession:
    """Create (or fetch the existing) SparkSession for this pipeline.

    Parameters
    ----------
    app_name:
        Name shown in the Spark UI / logs.
    shuffle_partitions:
        `spark.sql.shuffle.partitions` controls how many tasks a shuffle
        (groupBy, join, window function, etc.) fans out into. The Spark
        default (200) is tuned for large clusters; for a dataset the size
        of this case study (thousands to low millions of rows) that just
        creates thousands of tiny, mostly-empty tasks and slows things
        down. A small number here is a deliberate choice for this
        dataset's scale, not an oversight -- bump it back up (or leave it
        unset) once real data volumes justify more parallelism.

    Returns
    -------
    An active SparkSession, created once and reused via
    ``SparkSession.builder.getOrCreate()``.
    """
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory for `path` if it doesn't exist yet.

    `write_yaml`/`write_json` already do this internally, but
    `matplotlib.figure.Figure.savefig` does not -- it raises
    `FileNotFoundError` if the target directory doesn't exist. Call this
    before any `savefig` to a path under a directory that might not have
    been created yet (e.g. a fresh `reports/` on a clean checkout).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    """Read a YAML config file into a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_yaml(path: str, payload: Dict[str, Any]) -> None:
    """Write a dict to YAML, creating parent directories as needed.

    Used for metrics, EDA summaries, and fairness reports -- all of which
    are small, driver-side dicts by the time they reach this function
    (never full Spark DataFrames), so plain YAML is a reasonable,
    human-readable format for them.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(_make_yaml_safe(payload), f, sort_keys=False)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write a dict to JSON, creating parent directories as needed."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(_make_yaml_safe(payload), f, indent=2, default=str)


def _make_yaml_safe(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalar types to native Python types.

    YAML's safe dumper doesn't know how to serialise numpy.float64,
    numpy.int64, etc., which show up constantly once anything upstream
    (e.g. a scipy test statistic) touches numpy. Rather than sprinkling
    `float(...)` casts at every call site, funnel everything through here
    once before writing.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: _make_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_yaml_safe(v) for v in obj]
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj
