"""
Population Stability Index (PSI) drift monitoring, computed entirely in
Spark SQL so it can run as a scheduled batch job against however much
production traffic has landed (not just what fits in one process' memory).

PSI formula (standard binned version):
    PSI = sum_i (cur_pct_i - ref_pct_i) * ln(cur_pct_i / ref_pct_i)

Thresholds (industry-standard rule of thumb):
    PSI < 0.10            -> OK
    0.10 <= PSI <= 0.25    -> WATCH
    PSI > 0.25             -> DRIFT   (drift_flag = true)
"""
from __future__ import annotations
import math 

from dataclasses import dataclass

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

WATCH_THRESHOLD = 0.10
DRIFT_THRESHOLD = 0.25
EPS = 1e-6


@dataclass
class PsiThresholds:
    watch: float = WATCH_THRESHOLD
    drift: float = DRIFT_THRESHOLD


def _status(psi: float, thresholds: PsiThresholds) -> str:
    if psi > thresholds.drift:
        return "DRIFT"
    if psi >= thresholds.watch:
        return "WATCH"
    return "OK"


def compute_psi_spark(
    reference_df: DataFrame,
    current_df: DataFrame,
    column: str,
    n_bins: int = 10,
) -> float:
    """Bins `column` from `reference_df` into `n_bins` equal-width buckets, 
    and computes PSI with a single Spark SQL join.
    """
    spark = reference_df.sparkSession

    edges = reference_df.select(
        F.min(column).alias("lo"), F.max(column).alias("hi")
    ).collect()[0]
    lo, hi = float(edges["lo"]), float(edges["hi"])
    if hi == lo:
        # Constant reference column: only two outcomes exist, "same as the
        # reference value" or "different". Returning 0.0 here would hide a
        # shift in the current data (e.g. a flag that is always 0 in the
        # reference but 1 for part of the current slice).
        ref_total = reference_df.count()
        cur_total = current_df.count()
        cur_same = current_df.where(F.col(column) == F.lit(lo)).count()
        ref_pcts = [(ref_total + EPS) / (ref_total + 2 * EPS), EPS / (ref_total + 2 * EPS)]
        cur_pcts = [(cur_same + EPS) / (cur_total + 2 * EPS),
                    (cur_total - cur_same + EPS) / (cur_total + 2 * EPS)]
        return float(sum((c - r) * math.log(c / r) for c, r in zip(cur_pcts, ref_pcts)))
    width = (hi - lo) / n_bins

    def bucket_expr(df: DataFrame) -> DataFrame:
        bucket = F.least(
            F.greatest(
                F.floor((F.col(column) - F.lit(lo)) / F.lit(width)), F.lit(0)
            ),
            F.lit(n_bins - 1),
        )
        return df.withColumn("bucket", bucket.cast("int"))

    ref_counts = (
        bucket_expr(reference_df).groupBy("bucket").agg(F.count("*").alias("ref_n"))
    )
    cur_counts = (
        bucket_expr(current_df).groupBy("bucket").agg(F.count("*").alias("cur_n"))
    )
    ref_total = reference_df.count()
    cur_total = current_df.count()

    all_buckets = spark.createDataFrame([(i,) for i in range(n_bins)], ["bucket"])
    joined = (
        all_buckets.join(ref_counts, "bucket", "left")
        .join(cur_counts, "bucket", "left")
        .fillna(0, subset=["ref_n", "cur_n"])
        .withColumn("ref_pct", (F.col("ref_n") + F.lit(EPS)) / (F.lit(ref_total) + F.lit(n_bins * EPS)))
        .withColumn("cur_pct", (F.col("cur_n") + F.lit(EPS)) / (F.lit(cur_total) + F.lit(n_bins * EPS)))
        .withColumn("term", (F.col("cur_pct") - F.col("ref_pct")) * F.log(F.col("cur_pct") / F.col("ref_pct")))
    )
    psi = joined.agg(F.sum("term").alias("psi")).collect()[0]["psi"]
    return float(psi) if psi is not None else 0.0


def psi_report(
    reference_df: DataFrame,
    current_df: DataFrame,
    columns: list[str],
    n_bins: int = 10,
    thresholds: PsiThresholds = PsiThresholds(),
) -> DataFrame:
    """Runs compute_psi_spark for every column and returns a Spark DataFrame
    with one row per feature, a `drift_flag` boolean, and a status label —
    this is the artifact a monitoring job would write to a metrics table."""
    spark = reference_df.sparkSession
    rows = []
    for col in columns:
        psi = compute_psi_spark(reference_df, current_df, col, n_bins=n_bins)
        rows.append(
            (
                col,
                round(psi, 6),
                psi > thresholds.drift,
                _status(psi, thresholds),
            )
        )
    return spark.createDataFrame(rows, ["feature", "psi", "drift_flag", "status"]).orderBy(
        F.desc("psi")
    )


def score_psi_flag(
    reference_scores: DataFrame,
    current_scores: DataFrame,
    score_column: str = "fraud_probability",
    n_bins: int = 10,
    thresholds: PsiThresholds = PsiThresholds(),
) -> dict:
    """Convenience wrapper for the single most important drift signal in a
    fraud model: has the *output score* distribution shifted? Returns a
    plain dict (JSON-serializable) so it can be written straight into the
    monitoring artifact consumed by the FastAPI /drift endpoint."""
    psi = compute_psi_spark(reference_scores, current_scores, score_column, n_bins=n_bins)
    return {
        "metric": score_column,
        "psi": round(psi, 6),
        "drift_flag": psi > thresholds.drift,
        "status": _status(psi, thresholds),
        "watch_threshold": thresholds.watch,
        "drift_threshold": thresholds.drift,
    }


if __name__ == "__main__":
    import sys
    from spark_session import get_spark, quiet_spark
    from spark_etl import load_raw, schema_and_null_check, add_features, chronological_split, full_feature_list

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/creditcard.csv"
    spark = get_spark()
    quiet_spark(spark)

    raw = load_raw(spark, data_path)
    clean = schema_and_null_check(raw)
    featurized = add_features(clean)
    split = chronological_split(featurized)

    # Cache: otherwise every PSI call recomputes the whole chain
    # (read, dedupe, windows, sort) from scratch for each feature.
    split.val.cache()
    split.test.cache()

    # Sanity check: which hours of the dataset each slice covers.
    # This first action also fills the cache.
    for name, d in [("val", split.val), ("test", split.test)]:
        r = d.agg(F.count("*").alias("n"), F.sum("Class").alias("frauds"),
                  F.min("Time").alias("t0"), F.max("Time").alias("t1")).first()
        print(f"{name:<4}: {r.n:,} rows | {r.frauds} frauds | Time {r.t0/3600:.1f}h to {r.t1/3600:.1f}h")

    # Deterministic functions of the clock: consecutive slices differ by design,
    # so a flag on them says nothing about the model's inputs.
    TIME_DERIVED = {"hour_of_day", "hour_sin", "hour_cos", "is_night", "amount_x_hour"}
    monitored = [f for f in full_feature_list() if f not in TIME_DERIVED]
    report = psi_report(split.val, split.test, monitored)
    report.show(40, truncate=False)

    n_drift = report.where("drift_flag").count()
    n_watch = report.where("status = 'WATCH'").count()
    print(f"Features flagged DRIFT: {n_drift} | WATCH: {n_watch}")

    spark.stop()
