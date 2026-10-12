"""
Spark ETL for the credit-card fraud dataset.

Replaces the pandas ingestion / feature-engineering / EDA cells of the
original notebook with PySpark DataFrame ops + Spark SQL, so the pipeline
scales past a single machine's memory if the raw transaction log grows.

Feature engineering matches the original notebook 1:1:
  - hour_of_day, hour_sin, hour_cos, is_night   (cyclical time-of-day)
  - log_amount, amount_bucket, amount_x_hour    (amount transforms)
  - tx_velocity_1h, rolling_avg_amount          (trailing 1h window)

One deliberate simplification carried over from the source notebook: the
Kaggle dataset has no card/account id, so the "velocity" window is global
(ordered by Time only) rather than partitioned by cardholder. In a real
deployment this window would be `Window.partitionBy("card_id").orderBy(...)`.
"""
from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

ENGINEERED_FEATURES = [
    "hour_of_day", "hour_sin", "hour_cos", "is_night",
    "log_amount", "amount_bucket", "amount_x_hour",
    "tx_velocity_1h", "rolling_avg_amount",
]
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
RAW_FEATURES = ["Time", "Amount"] + PCA_FEATURES
TARGET = "Class"


def load_raw(spark: SparkSession, path: str) -> DataFrame:
    """Read the raw transaction CSV with an explicit schema (no inferSchema
    pass — cheaper and avoids silent type drift on a 280k+ row file)."""
    from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

    schema = StructType(
        [StructField("Time", DoubleType(), False)]
        + [StructField(f"V{i}", DoubleType(), False) for i in range(1, 29)]
        + [
            StructField("Amount", DoubleType(), False),
            StructField("Class", IntegerType(), False),
        ]
    )
    return spark.read.csv(path, header=True, schema=schema)


def schema_and_null_check(df: DataFrame) -> DataFrame:
    """Prints a null/dup report via Spark SQL and returns a de-duplicated df."""
    df.createOrReplaceTempView("raw_tx")
    spark = df.sparkSession

    null_expr = ", ".join(f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS {c}_nulls" for c in df.columns)
    null_report = spark.sql(f"SELECT {null_expr} FROM raw_tx").collect()[0].asDict()
    total_nulls = sum(null_report.values())

    total = df.count()
    deduped = df.dropDuplicates()
    n_dupes = total - deduped.count()

    print(f"Rows: {total:,} | Null cells: {total_nulls:,} | Duplicate rows dropped: {n_dupes:,}")
    return deduped


def class_distribution(df: DataFrame) -> DataFrame:
    """Spark SQL class balance report."""
    df.createOrReplaceTempView("tx")
    return df.sparkSession.sql(
        """
        SELECT
          Class,
          COUNT(*) AS n,
          ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 4) AS pct
        FROM tx
        GROUP BY Class
        ORDER BY Class
        """
    )


def amount_stats_by_class(df: DataFrame) -> DataFrame:
    """Spark SQL descriptive stats on Amount, split by fraud/non-fraud."""
    df.createOrReplaceTempView("tx")
    return df.sparkSession.sql(
        """
        SELECT
          Class,
          COUNT(*)                                   AS n,
          ROUND(AVG(Amount), 2)                       AS mean_amount,
          ROUND(percentile_approx(Amount, 0.5), 2)     AS median_amount,
          ROUND(STDDEV(Amount), 2)                     AS stddev_amount,
          ROUND(MAX(Amount), 2)                        AS max_amount
        FROM tx
        GROUP BY Class
        ORDER BY Class
        """
    )


def add_features(df: DataFrame) -> DataFrame:
    """Adds the engineered feature set using a mix of Spark SQL (time/amount
    transforms) and the DataFrame window API (trailing 1h velocity)."""
    df.createOrReplaceTempView("tx_base")
    spark = df.sparkSession

    with_time_amount = spark.sql(
        """
        SELECT
          *,
          (Time % 86400) / 3600.0                                    AS hour_of_day,
          SIN(2 * PI() * ((Time % 86400) / 3600.0) / 24)              AS hour_sin,
          COS(2 * PI() * ((Time % 86400) / 3600.0) / 24)              AS hour_cos,
          CASE WHEN (Time % 86400) / 3600.0 < 6
                 OR (Time % 86400) / 3600.0 >= 22 THEN 1 ELSE 0 END   AS is_night,
          LOG1P(Amount)                                               AS log_amount,
          NTILE(5) OVER (ORDER BY Amount) - 1                         AS amount_bucket
        FROM tx_base
        """
    )
    with_time_amount = with_time_amount.withColumn(
        "amount_x_hour", F.col("log_amount") * F.col("hour_of_day")
    )

    # Trailing 1-hour window ordered by Time (see module docstring re: no
    # card-level partition key in this dataset).
    trailing_1h = Window.orderBy(F.col("Time").cast("long")).rangeBetween(-3600, 0)
    windowed = with_time_amount.withColumn(
        "tx_velocity_1h", F.count("Amount").over(trailing_1h)
    ).withColumn(
        "rolling_avg_amount", F.avg("Amount").over(trailing_1h)
    )

    return windowed.orderBy("Time")


def engineered_feature_correlations(df: DataFrame) -> DataFrame:
    """Spark SQL corr() of each engineered feature against the label,
    unpivoted into one row per feature so it's easy to bar-chart."""
    df.createOrReplaceTempView("featurized")
    spark = df.sparkSession
    corr_exprs = ", ".join(f"corr({f}, Class) AS {f}" for f in ENGINEERED_FEATURES)
    wide = spark.sql(f"SELECT {corr_exprs} FROM featurized").collect()[0].asDict()
    rows = [(feat, float(val) if val is not None else None) for feat, val in wide.items()]
    return spark.createDataFrame(rows, ["feature", "corr_with_class"]).orderBy("corr_with_class")


@dataclass
class SplitResult:
    train: DataFrame
    val: DataFrame
    test: DataFrame


def chronological_split(df: DataFrame, train_frac: float = 0.70, val_frac: float = 0.85) -> SplitResult:
    """Time-ordered 70/15/15 split via a row_number() window — mirrors the
    original notebook's `.iloc` split but expressed in Spark SQL so it
    stays correct under repartitioning (iloc-by-position isn't safe once
    a DataFrame is distributed)."""
    df.createOrReplaceTempView("featurized_ordered")
    spark = df.sparkSession
    ranked = spark.sql(
        """
        SELECT *, ROW_NUMBER() OVER (ORDER BY Time) AS rn, COUNT(*) OVER () AS n
        FROM featurized_ordered
        """
    )
    train_end = F.floor(F.col("n") * F.lit(train_frac))
    val_end = F.floor(F.col("n") * F.lit(val_frac))

    train = ranked.where(F.col("rn") <= train_end).drop("rn", "n")
    val = ranked.where((F.col("rn") > train_end) & (F.col("rn") <= val_end)).drop("rn", "n")
    test = ranked.where(F.col("rn") > val_end).drop("rn", "n")
    return SplitResult(train=train, val=val, test=test)


# Excluded from the model on purpose:
#  - Time: seconds since the first transaction (a collection artifact; trees can't
#    extrapolate past the training range, and it triggers artificial drift flags)
#  - tx_velocity_1h, rolling_avg_amount: all-card traffic, not per-card behavior
MODEL_FEATURES = (
    ["Amount"] + PCA_FEATURES
    + [f for f in ENGINEERED_FEATURES if f not in ("tx_velocity_1h", "rolling_avg_amount")]
)

def full_feature_list() -> list[str]:
    return MODEL_FEATURES


if __name__ == "__main__":
    import sys
    from spark_session import get_spark, quiet_spark

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/creditcard_sample.csv"
    spark = get_spark()
    quiet_spark(spark)
    raw = load_raw(spark, data_path)
    clean = schema_and_null_check(raw)

    print("\n=== Class distribution ===")
    class_distribution(clean).show()

    print("=== Amount stats by class ===")
    amount_stats_by_class(clean).show()

    featurized = add_features(clean)
    print(f"Added {len(ENGINEERED_FEATURES)} engineered features. "
          f"Total columns: {len(featurized.columns)}")
    featurized.select(*ENGINEERED_FEATURES).describe().show()

    print("=== Engineered feature correlation with Class ===")
    engineered_feature_correlations(featurized).show(20, truncate=False)

    split = chronological_split(featurized)
    for name, d in [("train", split.train), ("val", split.val), ("test", split.test)]:
        n = d.count()
        f = d.filter("Class = 1").count()
        print(f"{name:<5}: {n:>8,} rows | {f:>4} frauds ({100 * f / n:.3f}%)")

    spark.stop()
