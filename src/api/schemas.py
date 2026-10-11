from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Transaction(BaseModel):
    """Raw transaction fields, matching the Kaggle creditcard.csv schema."""

    time: float = Field(..., description="Seconds elapsed since the first transaction in the dataset/session")
    amount: float = Field(..., ge=0, description="Transaction amount in USD")
    v: list[float] = Field(..., min_length=28, max_length=28, description="PCA components V1..V28, in order")

    # Optional trailing-1h context. In production this comes from a feature
    # store (e.g. a Redis/Flink job keyed by card id) populated in real
    # time; the batch Spark job computes the equivalent window offline for
    # training. Without it, velocity/rolling-avg default to "first seen".
    recent_amounts_1h: Optional[list[float]] = Field(
        default=None,
        description="Amounts of this card's other transactions in the trailing 1h window, if known",
    )

    @field_validator("v")
    @classmethod
    def check_v_length(cls, v):
        if len(v) != 28:
            raise ValueError("v must contain exactly 28 values (V1..V28)")
        return v


class PredictionResponse(BaseModel):
    fraud_probability: float
    fraud_flag: bool
    action: str  # "allow" | "review" | "block"
    threshold: float
    model_name: str


class DriftFeature(BaseModel):
    feature: str
    psi: float
    drift_flag: bool
    status: str


class DriftResponse(BaseModel):
    reference: str
    current: str
    n_features_drift: int
    n_features_watch: int
    score_drift: dict
    feature_drift: list[DriftFeature]
