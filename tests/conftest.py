from __future__ import annotations

from pyspark.sql import SparkSession

# Create a global variable for the Spark session
spark = None

# Hook that runs once when the test session starts


def pytest_sessionstart(session):
    global spark
    # Initialize SparkSession
    spark = SparkSession.builder.getOrCreate()


# Hook that runs once when the test session finishes


def pytest_sessionfinish(session, exitstatus):
    global spark
    # Stop the SparkSession
    if spark:
        spark.stop()
        spark = None
