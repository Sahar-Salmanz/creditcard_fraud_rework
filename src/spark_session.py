"""
Shared SparkSession factory for the fraud detection pipeline.
Local mode by default (spark.master=local[*]); point SPARK_MASTER at a
cluster URL via env var when this needs to run outside a laptop/CI box.
"""
import os
import sys
from pyspark.sql import SparkSession


def get_spark(app_name: str = "credit-fraud-pipeline") -> SparkSession:
    # Make Spark's Python workers use the same interpreter as this script
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    master = os.environ.get("SPARK_MASTER", "local[*]")
    return (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "8"))
        .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "2g"))
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def quiet_spark(spark: SparkSession) -> None:
    """Drop Spark's log level to WARN/ERROR so notebook/CLI output stays readable."""
    spark.sparkContext.setLogLevel("ERROR")
