from __future__ import annotations

import uuid

import pyspark.sql.functions as F
import pyspark.sql.types as T
import pytest

from pyspark_toolkit.uuid import uuid5


def test_uuid5_single_column(spark):
    """
    Test UUID5 generation with a single column matches Python's uuid.uuid5.
    """
    # when I have a single column of data
    data = [
        ("alice",),
        ("bob",),
        ("charlie",),
    ]
    df = spark.createDataFrame(data, ["name"])

    # and I generate UUID5 with OID namespace
    df = df.withColumn("uuid5_spark", uuid5("name", namespace=uuid.NAMESPACE_OID))

    # and I generate the same UUID5 with Python
    rows = df.collect()
    for row in rows:
        python_uuid = str(uuid.uuid5(uuid.NAMESPACE_OID, row["name"]))
        assert row["uuid5_spark"] == python_uuid


def test_uuid5_multiple_columns(spark):
    """
    Test UUID5 generation with multiple columns concatenated with separator.
    """
    # when I have multiple columns
    data = [
        ("alice", "smith", 30),
        ("bob", "jones", 25),
        ("charlie", "brown", 35),
    ]
    df = spark.createDataFrame(data, ["first_name", "last_name", "age"])

    # and I generate UUID5 with multiple columns and default separator
    df = df.withColumn("uuid5_spark", uuid5("first_name", "last_name", "age"))

    # and I generate the same UUID5 with Python
    rows = df.collect()
    for row in rows:
        # Concatenate with default separator "-"
        concatenated = f"{row['first_name']}-{row['last_name']}-{row['age']}"
        python_uuid = str(uuid.uuid5(uuid.NAMESPACE_OID, concatenated))
        assert row["uuid5_spark"] == python_uuid


def test_uuid5_custom_separator(spark):
    """
    Test UUID5 generation with custom separator.
    """
    # when I have multiple columns
    data = [
        ("alice", "smith"),
        ("bob", "jones"),
    ]
    df = spark.createDataFrame(data, ["first", "last"])

    # and I generate UUID5 with custom separator
    df = df.withColumn("uuid5_pipe", uuid5("first", "last", separator="|"))
    df = df.withColumn("uuid5_comma", uuid5("first", "last", separator=","))
    df = df.withColumn("uuid5_empty", uuid5("first", "last", separator=""))

    # and I verify with Python
    rows = df.collect()
    for row in rows:
        # Test pipe separator
        concatenated_pipe = f"{row['first']}|{row['last']}"
        python_uuid_pipe = str(uuid.uuid5(uuid.NAMESPACE_OID, concatenated_pipe))
        assert row["uuid5_pipe"] == python_uuid_pipe

        # Test comma separator
        concatenated_comma = f"{row['first']},{row['last']}"
        python_uuid_comma = str(uuid.uuid5(uuid.NAMESPACE_OID, concatenated_comma))
        assert row["uuid5_comma"] == python_uuid_comma

        # Test empty separator
        concatenated_empty = f"{row['first']}{row['last']}"
        python_uuid_empty = str(uuid.uuid5(uuid.NAMESPACE_OID, concatenated_empty))
        assert row["uuid5_empty"] == python_uuid_empty


def test_uuid5_different_namespaces(spark):
    """
    Test UUID5 generation with different namespace values.
    """
    # when I have a column
    data = [("test_value",)]
    df = spark.createDataFrame(data, ["value"])

    # and I generate UUID5 with different namespaces
    df = df.withColumn("uuid5_oid", uuid5("value", namespace=uuid.NAMESPACE_OID))
    df = df.withColumn("uuid5_dns", uuid5("value", namespace=uuid.NAMESPACE_DNS))
    df = df.withColumn("uuid5_url", uuid5("value", namespace=uuid.NAMESPACE_URL))
    df = df.withColumn("uuid5_x500", uuid5("value", namespace=uuid.NAMESPACE_X500))

    # and I verify each matches Python's implementation
    row = df.first()
    assert row["uuid5_oid"] == str(uuid.uuid5(uuid.NAMESPACE_OID, "test_value"))
    assert row["uuid5_dns"] == str(uuid.uuid5(uuid.NAMESPACE_DNS, "test_value"))
    assert row["uuid5_url"] == str(uuid.uuid5(uuid.NAMESPACE_URL, "test_value"))
    assert row["uuid5_x500"] == str(uuid.uuid5(uuid.NAMESPACE_X500, "test_value"))

    # and all UUIDs should be different
    uuids = [row["uuid5_oid"], row["uuid5_dns"], row["uuid5_url"], row["uuid5_x500"]]
    assert len(set(uuids)) == 4


def test_uuid5_with_null_values(spark):
    """
    Test UUID5 generation handles null values with placeholder.
    """
    # when I have data with nulls
    data = [
        ("alice", "smith"),
        ("bob", None),
        (None, "jones"),
        (None, None),
    ]
    df = spark.createDataFrame(data, ["first", "last"])

    # and I generate UUID5 with default null placeholder
    df = df.withColumn("uuid5_spark", uuid5("first", "last"))

    # and I verify with Python using null byte placeholder
    rows = df.collect()
    for row in rows:
        first = row["first"] if row["first"] is not None else "\0"
        last = row["last"] if row["last"] is not None else "\0"
        concatenated = f"{first}-{last}"
        python_uuid = str(uuid.uuid5(uuid.NAMESPACE_OID, concatenated))
        assert row["uuid5_spark"] == python_uuid


def test_uuid5_custom_null_placeholder(spark):
    """
    Test UUID5 generation with custom null placeholder.
    """
    # when I have data with nulls
    data = [
        ("alice", None),
        (None, "smith"),
    ]
    df = spark.createDataFrame(data, ["first", "last"])

    # and I generate UUID5 with custom null placeholder
    df = df.withColumn("uuid5_null", uuid5("first", "last", null_placeholder="NULL"))

    # and I verify with Python
    rows = df.collect()
    for row in rows:
        first = row["first"] if row["first"] is not None else "NULL"
        last = row["last"] if row["last"] is not None else "NULL"
        concatenated = f"{first}-{last}"
        python_uuid = str(uuid.uuid5(uuid.NAMESPACE_OID, concatenated))
        assert row["uuid5_null"] == python_uuid


def test_uuid5_with_column_objects(spark):
    """
    Test UUID5 generation accepts Column objects directly.
    """
    # when I have a dataframe
    data = [
        ("alice", "smith"),
        ("bob", "jones"),
    ]
    df = spark.createDataFrame(data, ["first", "last"])

    # and I pass Column objects instead of strings
    df = df.withColumn("uuid5_cols", uuid5(F.col("first"), F.col("last")))

    # and I compare with string column names
    df = df.withColumn("uuid5_strings", uuid5("first", "last"))

    # then both approaches should produce the same result
    rows = df.collect()
    for row in rows:
        assert row["uuid5_cols"] == row["uuid5_strings"]


def test_uuid5_with_numeric_columns(spark):
    """
    Test UUID5 generation with numeric columns that get cast to strings.
    """
    # when I have numeric columns
    data = [
        (1, 100.5),
        (2, 200.75),
        (3, 300.0),
    ]
    df = spark.createDataFrame(data, ["id", "amount"])

    # and I generate UUID5 with numeric columns
    df = df.withColumn("uuid5_spark", uuid5("id", "amount"))

    # and I verify with Python
    rows = df.collect()
    for row in rows:
        # Numbers are converted to strings in concatenation
        concatenated = f"{row['id']}-{row['amount']}"
        python_uuid = str(uuid.uuid5(uuid.NAMESPACE_OID, concatenated))
        assert row["uuid5_spark"] == python_uuid


def test_uuid5_no_columns_raises_error(spark):
    """
    Test that uuid5 raises ValueError when no columns are provided.
    """
    # when I have a dataframe
    data = [("test",)]
    df = spark.createDataFrame(data, ["value"])

    # and I call uuid5 without columns
    # then I should get a ValueError
    with pytest.raises(ValueError) as excinfo:
        df.withColumn("uuid5", uuid5())

    assert "No columns passed!" in str(excinfo.value)


def test_uuid5_deterministic(spark):
    """
    Test that UUID5 generation is deterministic for the same input.
    """
    # when I have duplicate values
    data = [
        ("alice", "smith"),
        ("bob", "jones"),
        ("alice", "smith"),  # duplicate
        ("charlie", "brown"),
        ("bob", "jones"),  # duplicate
    ]
    df = spark.createDataFrame(data, ["first", "last"])

    # and I generate UUID5
    df = df.withColumn("uuid5", uuid5("first", "last"))

    # then duplicate inputs should produce identical UUIDs
    rows = df.collect()
    assert rows[0]["uuid5"] == rows[2]["uuid5"]  # alice smith
    assert rows[1]["uuid5"] == rows[4]["uuid5"]  # bob jones
    assert rows[3]["uuid5"] != rows[0]["uuid5"]  # charlie != alice


def test_uuid5_format_compliance(spark):
    """
    Test that generated UUIDs comply with UUID version 5 format.
    """
    # when I generate UUID5s
    data = [
        ("test1",),
        ("test2",),
        ("test3",),
    ]
    df = spark.createDataFrame(data, ["value"])
    df = df.withColumn("uuid5", uuid5("value"))

    # then all UUIDs should have correct format
    rows = df.collect()
    for row in rows:
        uuid_str = row["uuid5"]

        # Check format: 8-4-4-4-12 hexadecimal digits
        parts = uuid_str.split("-")
        assert len(parts) == 5
        assert len(parts[0]) == 8
        assert len(parts[1]) == 4
        assert len(parts[2]) == 4
        assert len(parts[3]) == 4
        assert len(parts[4]) == 12

        # Check version field (should start with '5')
        assert parts[2][0] == "5"

        # Verify it's a valid UUID by parsing with Python's uuid module
        parsed = uuid.UUID(uuid_str)
        assert parsed.version == 5


def test_uuid5_large_dataset_performance(spark):
    """
    Test UUID5 generation works efficiently on larger datasets.
    """
    # when I have a larger dataset
    data = [(f"user_{i}", f"domain_{i % 10}", i % 100) for i in range(1000)]
    df = spark.createDataFrame(data, ["username", "domain", "score"])

    # and I generate UUID5s
    df = df.withColumn("uuid5", uuid5("username", "domain", "score"))

    # then all UUIDs should be generated and unique for unique inputs
    result = df.select("uuid5").distinct().count()
    # We have 1000 rows but only 100 unique combinations (10 domains * 100 scores)
    # Actually each username is unique, so we should have 1000 unique UUIDs
    assert result == 1000

    # and spot check a few for correctness
    sample = df.limit(5).collect()
    for row in sample:
        concatenated = f"{row['username']}-{row['domain']}-{row['score']}"
        python_uuid = str(uuid.uuid5(uuid.NAMESPACE_OID, concatenated))
        assert row["uuid5"] == python_uuid


def test_uuid5_with_special_characters(spark):
    """
    Test UUID5 generation with special characters and Unicode.
    """
    # when I have data with special characters
    data = [
        ("alice@example.com", "pass#word!"),
        ("user-123", "value|pipe"),
        ("测试", "テスト"),  # Chinese and Japanese
        ("😀emoji", "test"),
    ]
    df = spark.createDataFrame(data, ["user", "value"])

    # and I generate UUID5
    df = df.withColumn("uuid5_spark", uuid5("user", "value"))

    # and I verify with Python
    rows = df.collect()
    for row in rows:
        concatenated = f"{row['user']}-{row['value']}"
        python_uuid = str(uuid.uuid5(uuid.NAMESPACE_OID, concatenated))
        assert row["uuid5_spark"] == python_uuid
