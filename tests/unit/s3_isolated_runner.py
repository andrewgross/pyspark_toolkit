#!/usr/bin/env python
"""
Isolated test runner for s3 module tests that may hang.

This script is designed to be run as a separate process that can be
completely terminated if it hangs. Run with:
    python tests/unit/test_s3_isolated.py
"""

from __future__ import annotations

import sys
import warnings

import pyspark.sql.functions as F
from pyspark.sql import SparkSession


def test_s3_signature_key():
    """Test the s3._get_signature_key function that is known to hang."""
    print("Testing s3._get_signature_key (expected to hang)...")

    # Create Spark session
    spark = SparkSession.builder.appName("s3_isolated_test").getOrCreate()

    try:
        # Import the problematic module
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from pyspark_utils.s3 import _get_signature_key

        # Test data
        data = [
            (b"test_secret_key", "20240101", "us-east-1", "s3"),
        ]
        df = spark.createDataFrame(data, ["secret_key", "date_stamp", "region", "service"])

        # This should hang due to deep HMAC call graph
        print("Attempting to generate signature key (this will hang)...")
        df = df.withColumn(
            "signing_key",
            _get_signature_key(F.col("secret_key"), F.col("date_stamp"), F.col("region"), F.col("service")),
        )

        # Force evaluation - this should hang
        result = df.collect()

        # If we get here, the issue has been fixed!
        print("SUCCESS: The s3._get_signature_key function completed!")
        print("The deep call graph issue may be resolved. Consider re-enabling the s3 module.")
        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    finally:
        try:
            spark.stop()
        except:
            pass


if __name__ == "__main__":
    sys.exit(test_s3_signature_key())
