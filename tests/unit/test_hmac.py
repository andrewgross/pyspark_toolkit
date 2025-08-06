from __future__ import annotations

import pyspark.sql.functions as F
import pyspark.sql.types as T

from pyspark_utils.hmac import _prepare_key, hmac_sha256
from tests.helpers import hmac_python


def test_hmac_short_key(spark):
    """
    Test that we can compute an HMAC with a short (<64 bits) key
    """
    key = b"e179017a-62b0-4996-8a38-e91aa9f1e179017a-62b0-4996-8aaaaaaaaaaa"
    message = b"foobar"
    data = [
        (key, message),
    ]
    columns = ["key", "message"]
    df = spark.createDataFrame(data, columns)
    df = df.withColumn("hmac", hmac_sha256(F.col("key"), F.col("message")))
    expected_result = hmac_python(key, message)
    assert expected_result == bytes(df.collect()[0]["hmac"]).hex()


def test_hmac_boundary_key(spark):
    """
    Test that we can compute an HMAC with a boundary (=64 bits) key
    """
    key = b"e179017a-62b0-4996-8a38-e91aa9f1e179017a-62b0-4996-8aaaaaaaaaaaa"
    message = b"foobar"
    data = [
        (key, message),
    ]
    columns = ["key", "message"]
    df = spark.createDataFrame(data, columns)
    df = df.withColumn("hmac", hmac_sha256(F.col("key"), F.col("message")))
    expected_result = hmac_python(key, message)
    assert expected_result == bytes(df.collect()[0]["hmac"]).hex()


def test_hmac_long_key(spark):
    """
    Test that we can compute the HMAC of a message with a long (>64 bits) key
    """
    key = b"e179017a-62b0-4996-8a38-e91aa9f1e179017a-62b0-4996-8aaaaaaaaaaaaaaaaaaaaaa"
    message = b"foobar"
    data = [
        (key, message),
    ]
    columns = ["key", "message"]
    df = spark.createDataFrame(data, columns)
    df = df.withColumn("hmac", hmac_sha256(F.col("key"), F.col("message")))
    expected_result = hmac_python(key, message)
    assert expected_result == bytes(df.collect()[0]["hmac"]).hex()


def test_prepare_short_key(spark):
    """
    Test that we can pad a key to the correct length
    """
    key = b"e179017a-62b0-4996-8a38-e91aa9f1e179017a-62b0-4996-8aaaaaaaaaa"
    assert len(key) <= 64
    block_size = 64
    data = [
        (key),
    ]
    df = spark.createDataFrame(data, T.BinaryType())
    df = df.withColumn(
        "key",
        _prepare_key(
            F.col("value"),
            block_size=block_size,
        ),
    )
    expected_result = b"e179017a-62b0-4996-8a38-e91aa9f1e179017a-62b0-4996-8aaaaaaaaaa\x00\x00"
    assert expected_result == bytes(df.collect()[0]["key"])


def test_prepare_long_key(spark):
    """
    Test that we can pad a key to the correct length
    """
    key = b"e179017a-62b0-4996-8a38-e91aa9f1e179017a-62b0-4996-8aaaaaaaaaaaaaaaaaaaa"
    assert len(key) > 64
    block_size = 64
    data = [
        (key),
    ]
    df = spark.createDataFrame(data, T.BinaryType())
    df = df.withColumn(
        "key",
        _prepare_key(
            F.col("value"),
            block_size=block_size,
        ),
    )
    expected_result = b"W)nY\x1a%7!\xc5\x9e\x93\xcaI\xbd\xca5@{X\xdcx\xa2\x7f\xf31E\x03\xd3>\x15;'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"  # noqa
    assert expected_result == bytes(df.collect()[0]["key"])
