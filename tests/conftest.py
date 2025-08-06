from __future__ import annotations

import pytest
from pyspark.sql import SparkSession

# Create a global variable for the Spark session
_spark_session = None

# Hook that runs once when the test session starts


def pytest_sessionstart(session):
    global _spark_session
    # Initialize SparkSession
    _spark_session = SparkSession.builder.getOrCreate()


# Hook that runs once when the test session finishes


def pytest_sessionfinish(session, exitstatus):
    global _spark_session
    # Stop the SparkSession
    if _spark_session:
        _spark_session.stop()
        _spark_session = None


@pytest.fixture
def spark():
    """Provide the global Spark session to tests."""
    global _spark_session
    return _spark_session
