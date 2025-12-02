from __future__ import annotations

import pyspark
import pytest
from pyspark.sql import SparkSession

# Create a global variable for the Spark session
_spark_session = None

# Hook that runs once when the test session starts

SPARK_VERSION = tuple(int(x) for x in pyspark.__version__.split(".")[:2])


def pytest_sessionstart(session):
    global _spark_session
    # Initialize SparkSession
    _spark_session = SparkSession.builder.getOrCreate()
    _spark_session.sparkContext.setLogLevel("ERROR")


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


def pytest_configure(config):
    config.addinivalue_line("markers", "spark35_only: run this test only on Spark 3.5.x")
    config.addinivalue_line("markers", "spark40_only: run this test only on Spark 4.0.x")


def pytest_runtest_setup(item):
    """
    Skip tests based on the Spark version.
    """

    spark35 = item.get_closest_marker("spark35_only")
    spark40 = item.get_closest_marker("spark40_only")

    if spark35 and SPARK_VERSION[0:2] != (3, 5):
        pytest.skip("spark35_only test, skipping on Spark %s" % (pyspark.__version__,))
    if spark40 and SPARK_VERSION[0:2] != (4, 0):
        pytest.skip("spark40_only test, skipping on Spark %s" % (pyspark.__version__,))
