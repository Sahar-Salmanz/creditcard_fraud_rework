"""
Real-time mirror of src/spark_etl.py::add_features for exactly one
transaction. Kept as plain numpy (no Spark) deliberately — starting a
SparkSession per API request would dominate p99 latency; Spark's job in
this project is the offline/batch ETL + drift monitoring, not point lookups.

tx_velocity_1h / rolling_avg_amount rely on the caller supplying
`recent_amounts_1h` (see schemas.Transaction) since a single request has no
visibility into the cardholder's recent history on its own. In production
that context would come from a low-latency feature store kept warm by a
streaming job; here it defaults to "this is the only transaction seen so
far in the window" when omitted, which is a documented approximation, not
a hidden one.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

PCA_FEATURES = [f"V{i}" for i in range(1, 29)]


def engineer_transaction(
    time: float,
    amount: float,
    v: list[float],
    recent_amounts_1h: Optional[list[float]] = None,
) -> dict:
    hour_of_day = (time % 86400) / 3600.0
    hour_sin = math.sin(2 * math.pi * hour_of_day / 24)
    hour_cos = math.cos(2 * math.pi * hour_of_day / 24)
    is_night = 1 if (hour_of_day < 6 or hour_of_day >= 22) else 0

    log_amount = math.log1p(amount)
    # amount_bucket needs a reference distribution (it's a quintile rank in
    # training); without the batch population here we fall back to the
    # middle bucket, which is the safest "unknown" default and matches the
    # near-zero correlation this feature had with the label in training.
    amount_bucket = 2.0
    amount_x_hour = log_amount * hour_of_day

    window = (recent_amounts_1h or []) + [amount]
    tx_velocity_1h = float(len(window))
    rolling_avg_amount = float(np.mean(window))

    features = {"Time": time, "Amount": amount}
    features.update({name: val for name, val in zip(PCA_FEATURES, v)})
    features.update(
        {
            "hour_of_day": hour_of_day,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "is_night": is_night,
            "log_amount": log_amount,
            "amount_bucket": amount_bucket,
            "amount_x_hour": amount_x_hour,
            "tx_velocity_1h": tx_velocity_1h,
            "rolling_avg_amount": rolling_avg_amount,
        }
    )
    return features


def to_feature_vector(features: dict, feature_order: list[str]) -> np.ndarray:
    return np.array([[features[f] for f in feature_order]], dtype=float)
