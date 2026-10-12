"""
Trains the fraud models on Spark-engineered features and writes everything
the FastAPI service needs to serve predictions and report drift.

Architecture note (why Spark stops at the feature table): SMOTE and
scikit-learn / XGBoost have no first-class Spark-native equivalents worth the
complexity at ~280k rows, so the already feature-engineered train / val / test
Spark DataFrames are collected to pandas with toPandas() and trained from
there. Spark's job is the ETL and the drift monitoring, the parts that
genuinely benefit from scaling out.

Training protocol (and why):
  1. Chronological split (done in Spark): train = earliest 70%, val = next
     15%, test = last 15%. Training on the past and scoring the future is
     what deployment looks like.
  2. The scaler and any resampling are fit on TRAIN only.
  3. ONE imbalance strategy per run, never stacked: class weights, or SMOTE-NC
     at 2% or 10% fraud. Stacking SMOTE on top of class weights compensates
     for the imbalance twice and distorts the probabilities.
  4. Every (strategy, model) pair is scored on VALIDATION; the winner is the
     highest validation PR-AUC.
  5. The decision threshold maximizes NET SAVINGS on validation
     (TP * COST_FN - FP * COST_FP), not F1: a miss costs far more than a
     false alarm.
  6. TEST is touched once, for the winner only, and reported with bootstrap
     95% intervals (the test split has only ~50 frauds, so single numbers
     are noisy).

Outputs (all under artifacts/):
  model.joblib            winning model
  scaler.joblib           StandardScaler fit on train
  metadata.json           feature order, scaled columns, thresholds, costs
  validation_results.json every (strategy, model) result on validation
  test_results.json       the winner on test, with bootstrap intervals
  drift_report.json       feature-level + score-level PSI (val=reference, test=current)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from spark_session import get_spark, quiet_spark
from spark_etl import (
    load_raw,
    schema_and_null_check,
    add_features,
    chronological_split,
    full_feature_list,
    TARGET,
)
from drift import psi_report, score_psi_flag

SEED = 42
COST_FN = 120.0   # $ cost of a missed fraud (false negative)
COST_FP = 5.0     # $ cost of a false alarm (false positive)

# Continuous columns that get standardized (only those present in the feature list are used).
SCALE_COLS = ["Amount", "log_amount", "hour_of_day", "hour_sin", "hour_cos", "amount_x_hour"]
# Discrete columns: scaled never, and SMOTE-NC copies them instead of interpolating them.
CATEGORICAL_COLS = ["is_night", "amount_bucket"]
# Pure functions of the clock. Consecutive time slices differ in them by construction,
# so a PSI flag on them says nothing about the model's inputs. (Same set as drift.py.)
TIME_DERIVED = {"hour_of_day", "hour_sin", "hour_cos", "is_night", "amount_x_hour"}

# One imbalance strategy per run. smote_ratio = target fraud count / legitimate count.
STRATEGIES = [
    {"name": "class_weights", "smote_ratio": None},
    {"name": "smote_2pct", "smote_ratio": 0.02},
    {"name": "smote_10pct", "smote_ratio": 0.10},
]

# Auto-block policy knobs (business decisions, not statistics: set them with the fraud team).
BLOCK_MIN_PRECISION = 0.90   # block only where validation precision is at least this
BLOCK_MIN_TP = 5             # ...and backed by at least this many true frauds
N_BOOTSTRAP = 500

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_DATA = PROJECT_ROOT / "data" / "creditcard.csv"


# metrics

def _cuts(y_true, y_score):
    """For every distinct score t: how many frauds (tp) and legit rows (fp)
    are flagged when we flag everything with score >= t."""
    order = np.argsort(-y_score, kind="stable")
    s, y = y_score[order], y_true[order]
    last = np.r_[s[1:] != s[:-1], True]          # last row of each run of tied scores
    return s[last], np.cumsum(y)[last], np.cumsum(1 - y)[last]


def best_net_savings_threshold(y_true, y_score):
    """Threshold that maximizes TP * COST_FN - FP * COST_FP, in one vectorized pass."""
    t, tp, fp = _cuts(y_true, y_score)
    net = tp * COST_FN - fp * COST_FP
    i = int(np.argmax(net))
    if net[i] <= 0:  # flagging anything loses money: set a threshold nothing reaches
        return float(np.nextafter(t[0], np.inf)), 0.0
    return float(t[i]), float(net[i])


def find_block_threshold(y_true, y_score, review_threshold):
    """Lowest score at or above the review threshold where validation precision
    is >= BLOCK_MIN_PRECISION with at least BLOCK_MIN_TP frauds. 1.01 = never auto-block."""
    t, tp, fp = _cuts(y_true, y_score)
    precision = tp / (tp + fp)
    ok = (precision >= BLOCK_MIN_PRECISION) & (tp >= BLOCK_MIN_TP) & (t >= review_threshold)
    return float(t[ok].min()) if ok.any() else 1.01


def evaluate(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 6),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "flag_rate": round(float((tp + fp) / len(y_true)), 5),   # share of traffic sent to review
        "net_savings_usd": float(tp * COST_FN - fp * COST_FP),
        "max_savings_usd": float(y_true.sum() * COST_FN),        # catch every fraud, flag nothing else
    }


def bootstrap_ci(y_true, y_score, threshold, n_boot=N_BOOTSTRAP, seed=SEED):
    """95% percentile intervals from resampling the test rows with replacement."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    out = {k: [] for k in ("pr_auc", "precision", "recall", "net_savings_usd")}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if yt.sum() == 0:
            continue
        pred = ys >= threshold
        tp = int((pred & (yt == 1)).sum())
        fp = int((pred & (yt == 0)).sum())
        fn = int(yt.sum()) - tp
        out["pr_auc"].append(average_precision_score(yt, ys))
        out["precision"].append(tp / (tp + fp) if tp + fp else np.nan)
        out["recall"].append(tp / (tp + fn))
        out["net_savings_usd"].append(tp * COST_FN - fp * COST_FP)
    return {k: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
            for k, v in out.items()}


# modeling

def resample(X, y, ratio, categorical_idx):
    """SMOTE-NC (or plain SMOTE if there are no discrete columns) on TRAIN only."""
    if ratio is None:
        return X, y
    n_legit, n_fraud = int((y == 0).sum()), int((y == 1).sum())
    target = int(n_legit * ratio)
    if target <= n_fraud or n_fraud < 2:
        return X, y
    kwargs = dict(sampling_strategy={1: target}, random_state=SEED, k_neighbors=min(5, n_fraud - 1))
    sampler = SMOTENC(categorical_features=categorical_idx, **kwargs) if categorical_idx else SMOTE(**kwargs)
    return sampler.fit_resample(X, y)


def build_models(use_weights: bool, pos_weight: float):
    cw = "balanced" if use_weights else None
    return {
        "Logistic Regression": LogisticRegression(
            C=0.01, max_iter=1000, class_weight=cw, solver="lbfgs", random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight=cw, n_jobs=-1, random_state=SEED
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight if use_weights else 1.0,
            eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0,
        ),
    }


def single_row_latency_ms(model, X, n=300):
    """p50 / p99 of predict_proba on one row at a time (what the API does per request)."""
    for i in range(20):
        model.predict_proba(X[i:i + 1])
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        model.predict_proba(X[i:i + 1])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(times, 50)), float(np.percentile(times, 99))


def main(data_path: str) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    spark = get_spark()
    quiet_spark(spark)

    # Spark: ETL, split, drift
    print("== Spark ETL ==")
    raw = load_raw(spark, data_path)
    clean = schema_and_null_check(raw)
    featurized = add_features(clean)
    split = chronological_split(featurized)
    features = full_feature_list()
    monitored = [f for f in features if f not in TIME_DERIVED]
    print(f"{len(features)} model features, {len(monitored)} monitored for drift")

    # Cache: otherwise every PSI call recomputes read -> dedupe -> windows -> sort.
    split.val.cache()
    split.test.cache()

    print("\n== PSI drift report (val = reference, test = current; time-of-day features excluded) ==")
    drift_df = psi_report(split.val, split.test, monitored)
    drift_df.show(10, truncate=False)
    drift_records = [row.asDict() for row in drift_df.collect()]

    print("== Collecting Spark-engineered features to pandas for training ==")
    cols = features + [TARGET]
    train_pd = split.train.select(*cols).toPandas()
    val_pd = split.val.select(*cols).toPandas()
    test_pd = split.test.select(*cols).toPandas()
    split.val.unpersist()
    split.test.unpersist()

    X_train = train_pd[features].to_numpy(dtype=float)
    y_train = train_pd[TARGET].to_numpy(dtype=int)
    X_val = val_pd[features].to_numpy(dtype=float)
    y_val = val_pd[TARGET].to_numpy(dtype=int)
    X_test = test_pd[features].to_numpy(dtype=float)
    y_test = test_pd[TARGET].to_numpy(dtype=int)
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        print(f"{name:<5}: {len(y):>8,} rows | {int(y.sum()):>4} frauds ({100 * y.mean():.3f}%)")

    # Scale (fit on train only)
    scale_cols = [c for c in SCALE_COLS if c in features]
    scale_idx = [features.index(c) for c in scale_cols]
    categorical_idx = [features.index(c) for c in CATEGORICAL_COLS if c in features]
    scaler = StandardScaler()
    X_train[:, scale_idx] = scaler.fit_transform(X_train[:, scale_idx])
    X_val[:, scale_idx] = scaler.transform(X_val[:, scale_idx])
    X_test[:, scale_idx] = scaler.transform(X_test[:, scale_idx])

    # Compare (strategy x model) on VALIDATION
    results = []
    best = None   # keep only the running winner in memory (unlimited-depth forests are big)
    for strat in STRATEGIES:
        X_fit, y_fit = resample(X_train, y_train, strat["smote_ratio"], categorical_idx)
        use_weights = strat["smote_ratio"] is None
        pos_weight = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))
        print(f"\n-- Strategy '{strat['name']}': {len(y_fit):,} training rows, {int(y_fit.sum()):,} fraud")
        for model_name, model in build_models(use_weights, pos_weight).items():
            t0 = time.time()
            model.fit(X_fit, y_fit)
            p_val = model.predict_proba(X_val)[:, 1]
            threshold, _ = best_net_savings_threshold(y_val, p_val)
            res = {"strategy": strat["name"], "model": model_name, **evaluate(y_val, p_val, threshold)}
            results.append(res)
            print(f"   {model_name:<20} val PR-AUC={res['pr_auc']:.4f}  net=${res['net_savings_usd']:>8,.0f}"
                  f"  precision={res['precision']:.3f} recall={res['recall']:.3f}  ({time.time() - t0:.0f}s)")
            if best is None or res["pr_auc"] > best["res"]["pr_auc"]:
                best = {"res": res, "model": model, "p_val": p_val, "threshold": threshold}

    # Winner: touch the test set ONCE
    model, threshold = best["model"], best["threshold"]
    block_threshold = find_block_threshold(y_val, best["p_val"], threshold)
    p_test = model.predict_proba(X_test)[:, 1]
    test_res = evaluate(y_test, p_test, threshold)
    test_res["bootstrap_95ci"] = bootstrap_ci(y_test, p_test, threshold)
    test_res.update(model=best["res"]["model"], strategy=best["res"]["strategy"],
                    selected_on="validation PR-AUC")

    print(f"\n== Winner (selected on validation): {test_res['model']} + {test_res['strategy']} ==")
    print(f"threshold={threshold:.4f} (review)  block_threshold={block_threshold:.4f}")
    print(f"TEST  precision={test_res['precision']:.3f}  recall={test_res['recall']:.3f}  "
          f"PR-AUC={test_res['pr_auc']:.3f}  TP={test_res['tp']} FP={test_res['fp']} FN={test_res['fn']}")
    print(f"TEST  net savings ${test_res['net_savings_usd']:,.0f} of a possible ${test_res['max_savings_usd']:,.0f}"
          f"  | {test_res['flag_rate'] * 10000:.0f} flagged per 10,000 transactions")
    print("TEST  95% intervals:", test_res["bootstrap_95ci"])

    # Score-level drift on the winner (same Spark session)
    print("\n== Score-level PSI (val vs test, winner) ==")
    val_scores = spark.createDataFrame([(float(s),) for s in best["p_val"]], ["fraud_probability"])
    test_scores = spark.createDataFrame([(float(s),) for s in p_test], ["fraud_probability"])
    score_drift = score_psi_flag(val_scores, test_scores)
    print(score_drift)
    spark.stop()

    # Save artifacts
    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    size_mb = os.path.getsize(ARTIFACTS_DIR / "model.joblib") / 1e6
    p50, p99 = single_row_latency_ms(model, X_val)
    print(f"\nmodel.joblib: {size_mb:.1f} MB | single-row predict_proba p50={p50:.1f} ms, p99={p99:.1f} ms")

    metadata = {
        "model_name": test_res["model"],
        "imbalance_strategy": test_res["strategy"],
        "features": features,
        "scale_columns": scale_cols,
        "scale_indices": scale_idx,
        "threshold": threshold,                 # flag / send to review at or above this
        "block_threshold": block_threshold,     # auto-block above this (1.01 = never)
        "cost_fn_usd": COST_FN,
        "cost_fp_usd": COST_FP,
        # Legacy keys the current main.py reads, set so its logic matches the new policy:
        # review if p >= threshold - 0.0, block if p > block_probability.
        "review_margin": 0.0,
        "block_probability": block_threshold,
    }
    (ARTIFACTS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (ARTIFACTS_DIR / "validation_results.json").write_text(json.dumps(results, indent=2))
    (ARTIFACTS_DIR / "test_results.json").write_text(json.dumps(test_res, indent=2))
    drift_artifact = {
        "reference": "validation split",
        "current": "test split",
        "excluded_features": sorted(TIME_DERIVED),
        "feature_drift": drift_records,
        "score_drift": score_drift,
        "n_features_drift": sum(1 for r in drift_records if r["drift_flag"]),
        "n_features_watch": sum(1 for r in drift_records if r["status"] == "WATCH"),
    }
    (ARTIFACTS_DIR / "drift_report.json").write_text(json.dumps(drift_artifact, indent=2))
    print(f"Artifacts written to {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_DATA))