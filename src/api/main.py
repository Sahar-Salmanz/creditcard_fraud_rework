"""
FastAPI service for the fraud model trained in src/train.py.

Run:
    uvicorn api.main:app --reload --port 8000   # from src/

Endpoints:
    POST /predict   score one transaction
    GET  /drift     latest offline PSI drift report (feature + score level)
    GET  /health    liveness/readiness + loaded model metadata
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException

from .features import engineer_transaction, to_feature_vector
from .schemas import DriftResponse, PredictionResponse, Transaction

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Serves fraud probability scores from a Spark-engineered, "
                 "scikit-learn-trained model, plus PSI drift status.",
    version="1.0.0",
)

_state: dict = {}


@app.on_event("startup")
def load_artifacts() -> None:
    try:
        _state["model"] = joblib.load(ARTIFACTS_DIR / "model.joblib")
        _state["scaler"] = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
        _state["metadata"] = json.loads((ARTIFACTS_DIR / "metadata.json").read_text())
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Missing model artifacts in {ARTIFACTS_DIR}. Run `python train.py <data_path>` first."
        ) from e
    # Drift report is monitoring output, not required for the service to boot.
    drift_path = ARTIFACTS_DIR / "drift_report.json"
    _state["drift_report"] = json.loads(drift_path.read_text()) if drift_path.exists() else None


@app.get("/health")
def health():
    meta = _state.get("metadata")
    if meta is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "model_name": meta["model_name"],
        "threshold": meta["threshold"],
        "n_features": len(meta["features"]),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(tx: Transaction):
    model = _state.get("model")
    scaler = _state.get("scaler")
    meta = _state.get("metadata")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw_features = engineer_transaction(
        time=tx.time, amount=tx.amount, v=tx.v, recent_amounts_1h=tx.recent_amounts_1h
    )
    X = to_feature_vector(raw_features, meta["features"])
    X[:, meta["scale_indices"]] = scaler.transform(X[:, meta["scale_indices"]])

    proba = float(model.predict_proba(X)[0, 1])
    threshold = meta["threshold"]
    margin = meta["review_margin"]
    block_p = meta["block_probability"]

    if proba > block_p:
        action = "block"
    elif proba >= threshold - margin:
        action = "review"
    else:
        action = "allow"

    return PredictionResponse(
        fraud_probability=round(proba, 6),
        fraud_flag=proba >= threshold,
        action=action,
        threshold=threshold,
        model_name=meta["model_name"],
    )


@app.get("/drift", response_model=DriftResponse)
def drift():
    report = _state.get("drift_report")
    if report is None:
        raise HTTPException(
            status_code=404,
            detail="No drift report available. Run `python train.py <data_path>` "
                   "(or a standalone `python drift.py`) to generate artifacts/drift_report.json.",
        )
    return DriftResponse(
        reference=report["reference"],
        current=report["current"],
        n_features_drift=report["n_features_drift"],
        n_features_watch=report["n_features_watch"],
        score_drift=report["score_drift"],
        feature_drift=report["feature_drift"],
    )
