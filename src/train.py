"""
Trains the fraud models on Spark-engineered features and writes everything
the FastAPI service needs to serve predictions + report drift.

Architecture note (why Spark stops here): SMOTE and scikit-learn/XGBoost
don't have first-class Spark-native equivalents worth the complexity for a
280k-row dataset, so this stage collects the *already feature-engineered*
train/val/test Spark DataFrames to pandas with toPandas() and trains as
before. Spark's job in this pipeline is the ETL/feature engineering and the
drift monitoring (both of which genuinely benefit from scaling out) — not
forcing every stage through Spark for its own sake.

Outputs (all under artifacts/):
  model.joblib          best model (by validation PR-AUC)
  scaler.joblib          StandardScaler fit on train
  metadata.json          feature list, scale columns, chosen threshold, cost params
  drift_report.json      feature-level + score-level PSI (val=reference, test=current)
  test_results.json      final held-out metrics for all three models
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

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
COST_FN = 120.0  # $ cost of a missed fraud (false negative)
COST_FP = 5.0    # $ cost of a false alarm (false positive)
SCALE_COLS = [
    "Time", "Amount", "log_amount", "hour_of_day", "hour_sin", "hour_cos",
    "amount_x_hour", "tx_velocity_1h", "rolling_avg_amount",
]
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def evaluate(y_true, y_proba, name="model", threshold=0.5, cost_fn=COST_FN, cost_fp=COST_FP):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "model": name,
        "threshold": round(float(threshold), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "net_savings_usd": round(float(tp * cost_fn - fp * cost_fp), 2),
    }

# finds the probability threshold that gives the highest F1 score
def find_best_threshold(y_true, y_proba):
    _, _, thresholds = precision_recall_curve(y_true, y_proba)
    best_t, best_score = 0.5, -np.inf
    for t in thresholds:
        score = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t, best_score


def main(data_path: str) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    spark = get_spark()
    quiet_spark(spark)

    print("== Spark ETL ==")
    raw = load_raw(spark, data_path)
    clean = schema_and_null_check(raw)
    featurized = add_features(clean)
    split = chronological_split(featurized)
    features = full_feature_list()

    print("\n== PSI drift report (val=reference vs test=current) ==")
    drift_df = psi_report(split.val, split.test, features)
    drift_df.show(40, truncate=False)
    drift_records = [row.asDict() for row in drift_df.collect()]

    print("\n== Collecting Spark-engineered features to pandas for training ==")
    cols = features + [TARGET]
    train_pd = split.train.select(*cols).toPandas()
    val_pd = split.val.select(*cols).toPandas()
    test_pd = split.test.select(*cols).toPandas()
    spark.stop()  # done with Spark once features are collected

    X_train_raw = train_pd[features].to_numpy(dtype=float)
    y_train = train_pd[TARGET].to_numpy(dtype=int)
    X_val_raw = val_pd[features].to_numpy(dtype=float)
    y_val = val_pd[TARGET].to_numpy(dtype=int)
    X_test_raw = test_pd[features].to_numpy(dtype=float)
    y_test = test_pd[TARGET].to_numpy(dtype=int)
    print(f"Train: {len(y_train):,} (fraud={y_train.sum()}) | "
          f"Val: {len(y_val):,} (fraud={y_val.sum()}) | "
          f"Test: {len(y_test):,} (fraud={y_test.sum()})")

    scale_idx = [features.index(c) for c in SCALE_COLS]
    scaler = StandardScaler()
    X_train_raw[:, scale_idx] = scaler.fit_transform(X_train_raw[:, scale_idx])
    X_val_raw[:, scale_idx] = scaler.transform(X_val_raw[:, scale_idx])
    X_test_raw[:, scale_idx] = scaler.transform(X_test_raw[:, scale_idx])

    n_legit = (y_train == 0).sum()
    n_fraud = max((y_train == 1).sum(), 1)
    smote_ratio = 0.10
    target_n_fraud = max(int(n_legit * smote_ratio), n_fraud)
    if target_n_fraud > n_fraud and n_fraud >= 2:
        smote = SMOTE(sampling_strategy={1: target_n_fraud}, random_state=SEED,
                       k_neighbors=min(5, n_fraud - 1))
        X_train, y_train_s = smote.fit_resample(X_train_raw, y_train)
        print(f"After SMOTE: fraud={y_train_s.sum():,} / total={len(y_train_s):,}")
    else:
        X_train, y_train_s = X_train_raw, y_train
        print("Skipping SMOTE (too few minority samples in this slice) — using raw training data.")

    pos_weight = (y_train_s == 0).sum() / max((y_train_s == 1).sum(), 1)
    models = {
        "Logistic Regression": LogisticRegression(
            C=0.01, max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, oob_score=True, random_state=SEED
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight,
            eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0,
        ),
    }

    print("\n== Training ==")
    trained = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train_s)
        trained[name] = model
        print(f"[{name}] trained in {time.time() - t0:.1f}s")

    print("\n== Threshold tuning on validation set ==")
    optimal_thresholds = {}
    val_results = {}
    for name, model in trained.items():
        proba = model.predict_proba(X_val_raw)[:, 1]
        t, f1 = find_best_threshold(y_val, proba)
        optimal_thresholds[name] = t
        val_results[name] = evaluate(y_val, proba, name=name, threshold=t)
        print(f"{name:<22} optimal threshold={t:.4f}  val F1={f1:.4f}")

    print("\n== Final test-set evaluation ==")
    test_results = []
    for name, model in trained.items():
        proba = model.predict_proba(X_test_raw)[:, 1]
        res = evaluate(y_test, proba, name=name, threshold=optimal_thresholds[name])
        test_results.append(res)
        print(res)

    best_name = max(test_results, key=lambda r: r["pr_auc"])["model"]
    best_model = trained[best_name]
    best_threshold = optimal_thresholds[best_name]
    print(f"\nBest model by test PR-AUC: {best_name} (threshold={best_threshold:.4f})")

    print("\n== Score-level PSI (val vs test, best model) ==")
    val_scores = best_model.predict_proba(X_val_raw)[:, 1]
    test_scores = best_model.predict_proba(X_test_raw)[:, 1]
    spark2 = get_spark()
    quiet_spark(spark2)
    val_scores_df = spark2.createDataFrame([(float(s),) for s in val_scores], ["fraud_probability"])
    test_scores_df = spark2.createDataFrame([(float(s),) for s in test_scores], ["fraud_probability"])
    score_drift = score_psi_flag(val_scores_df, test_scores_df)
    spark2.stop()
    print(score_drift)

    print("\n== Saving artifacts ==")
    joblib.dump(best_model, ARTIFACTS_DIR / "model.joblib")
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")

    metadata = {
        "model_name": best_name,
        "features": features,
        "scale_columns": SCALE_COLS,
        "scale_indices": scale_idx,
        "threshold": best_threshold,
        "cost_fn_usd": COST_FN,
        "cost_fp_usd": COST_FP,
        "review_margin": 0.15,
        "block_probability": 0.85,
    }
    (ARTIFACTS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    drift_artifact = {
        "reference": "validation split",
        "current": "test split",
        "feature_drift": drift_records,
        "score_drift": score_drift,
        "n_features_drift": sum(1 for r in drift_records if r["drift_flag"]),
        "n_features_watch": sum(1 for r in drift_records if r["status"] == "WATCH"),
    }
    (ARTIFACTS_DIR / "drift_report.json").write_text(json.dumps(drift_artifact, indent=2))
    (ARTIFACTS_DIR / "test_results.json").write_text(json.dumps(test_results, indent=2))

    print(f"Artifacts written to {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "data/creditcard_sample.csv")
