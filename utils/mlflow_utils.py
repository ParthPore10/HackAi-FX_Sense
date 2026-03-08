from __future__ import annotations
import os
from contextlib import contextmanager


def _enabled() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI") or os.environ.get("MLFLOW_ENABLE"))


@contextmanager
def mlflow_run(run_name: str):
    if not _enabled():
        yield None
        return
    try:
        import mlflow

        if os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "fxsense"))
        with mlflow.start_run(run_name=run_name) as run:
            yield mlflow
    except Exception:
        yield None


def log_dict(mlflow, d: dict, prefix: str = "metric"):
    if not mlflow:
        return
    for k, v in d.items():
        try:
            mlflow.log_metric(f"{prefix}.{k}", float(v))
        except Exception:
            continue