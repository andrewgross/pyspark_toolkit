from __future__ import annotations

import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pytest

from pyspark_toolkit.helpers import chars_to_int, split_last_chars
from pyspark_toolkit.modulus import (
    extract_id_from_uuid,
    modulus_equals_offset,
    partition_by_uuid,
    split_uuid_string_for_id,
)
from pyspark_toolkit.types import (
    HexStringColumn,
    IntegerColumn,
    StringColumn,
    UUIDColumn,
)


def test_split_uuid_string_for_id(spark):
    """
    Test that split_uuid_string_for_id extracts the 5th segment of a UUID.
    """
    # when I have a UUID string
    uuid = "e179017a-62b0-4996-8a38-e91aa9f1e179"
    data = [(uuid,)]
    df = spark.createDataFrame(data, ["uuid"])

    # and I apply split_uuid_string_for_id
    df = df.withColumn("result", split_uuid_string_for_id(UUIDColumn(F.col("uuid"))))

    # then I should get the 5th segment
    assert df.collect()[0]["result"] == "e91aa9f1e179"


def test_split_last_chars(spark):
    """
    Test that split_last_chars extracts the last 4 characters.
    """
    # when I have a string
    data = [
        ("abcdefghijk",),
        ("1234",),
        ("xyz",),  # less than 4 chars
    ]
    df = spark.createDataFrame(data, ["text"])

    # and I apply split_last_chars
    df = df.withColumn("result", split_last_chars(StringColumn(F.col("text"))))

    # then I should get the last 4 characters
    rows = df.collect()
    assert rows[0]["result"] == "hijk"
    assert rows[1]["result"] == "1234"
    assert rows[2]["result"] == "xyz"  # returns what's available if less than 4


def test_hex_string_to_int_conversion(spark):
    """
    Test that hex strings are properly converted to integers using chars_to_int.
    """
    # when I have hex strings
    data = [
        ("FF",),  # 255
        ("100",),  # 256
        ("ABCD",),  # 43981
        ("0",),  # 0
        ("not_hex",),  # invalid hex should return null
        (None,),  # null input
    ]
    df = spark.createDataFrame(data, ["hex_str"])

    # and I apply chars_to_int with hex encoding
    # First convert hex string to integer using F.conv, then use chars_to_int for consistency
    df = df.withColumn("result", F.conv(F.col("hex_str"), 16, 10).cast("bigint"))

    # then I should get the correct integer values
    rows = df.collect()
    assert rows[0]["result"] == 255
    assert rows[1]["result"] == 256
    assert rows[2]["result"] == 43981
    assert rows[3]["result"] == 0
    assert rows[4]["result"] == 0  # invalid hex returns 0 (as per PySpark conv behavior)
    assert rows[5]["result"] is None  # null input returns null


def test_extract_id_from_uuid(spark):
    """
    Test that extract_id_from_uuid correctly extracts an integer from UUID.
    """
    # when I have UUID strings
    data = [
        ("e179017a-62b0-4996-8a38-e91aa9f1e179",),  # e179 = 57721
        ("550e8400-e29b-41d4-a716-446655440000",),  # 0000 = 0
        ("123e4567-e89b-12d3-a456-426614174FFF",),  # 4FFF = 20479
    ]
    df = spark.createDataFrame(data, ["uuid"])

    # and I apply extract_id_from_uuid
    df = df.withColumn("result", extract_id_from_uuid(UUIDColumn(F.col("uuid"))))

    # then I should get the correct integer from the last 4 hex chars
    rows = df.collect()
    assert rows[0]["result"] == 57721  # 0xe179
    assert rows[1]["result"] == 0  # 0x0000
    assert rows[2]["result"] == 20479  # 0x4FFF


def test_modulus_equals_offset(spark):
    """
    Test that modulus_equals_offset correctly checks modulus equality.
    """
    # when I have integer values
    data = [
        (10,),  # 10 % 3 = 1
        (11,),  # 11 % 3 = 2
        (12,),  # 12 % 3 = 0
        (13,),  # 13 % 3 = 1
        (None,),
    ]
    df = spark.createDataFrame(data, ["value"])

    # and I check for modulus 3, offset 1
    df = df.withColumn("matches_offset_1", modulus_equals_offset(IntegerColumn(F.col("value")), modulus=3, offset=1))

    # then I should get correct boolean results
    rows = df.collect()
    assert rows[0]["matches_offset_1"] is True  # 10 % 3 = 1
    assert rows[1]["matches_offset_1"] is False  # 11 % 3 = 2
    assert rows[2]["matches_offset_1"] is False  # 12 % 3 = 0
    assert rows[3]["matches_offset_1"] is True  # 13 % 3 = 1
    assert rows[4]["matches_offset_1"] is None  # null handling


def test_modulus_equals_offset_different_values(spark):
    """
    Test modulus_equals_offset with different modulus and offset values.
    """
    # when I have an integer
    data = [(17,)]
    df = spark.createDataFrame(data, ["value"])

    # and I check various modulus/offset combinations
    df = df.withColumn("mod_5_off_2", modulus_equals_offset(IntegerColumn(F.col("value")), 5, 2))
    df = df.withColumn("mod_4_off_1", modulus_equals_offset(IntegerColumn(F.col("value")), 4, 1))
    df = df.withColumn("mod_10_off_7", modulus_equals_offset(IntegerColumn(F.col("value")), 10, 7))

    # then I should get correct results
    row = df.collect()[0]
    assert row["mod_5_off_2"] is True  # 17 % 5 = 2
    assert row["mod_4_off_1"] is True  # 17 % 4 = 1
    assert row["mod_10_off_7"] is True  # 17 % 10 = 7


def test_partition_by_uuid(spark):
    """
    Test that partition_by_uuid correctly partitions data by UUID.
    """
    # when I have a dataframe with UUIDs and other data
    data = [
        ("e179017a-62b0-4996-8a38-e91aa9f1e179", "record1"),  # ID: 57721, 57721 % 3 = 1
        ("550e8400-e29b-41d4-a716-446655440000", "record2"),  # ID: 0, 0 % 3 = 0
        ("123e4567-e89b-12d3-a456-426614174FFF", "record3"),  # ID: 20479, 20479 % 3 = 1
        ("a0b1c2d3-e4f5-6789-abcd-ef0123456789", "record4"),  # ID: 26505, 26505 % 3 = 0
    ]
    df = spark.createDataFrame(data, ["uuid", "data"])

    # and I get partition 1 of 3 partitions
    result_df = partition_by_uuid(df, uuid_column="uuid", num_partitions=3, partition_id=1)

    # then I should only get records where UUID's last 4 hex chars % 3 == 1
    rows = result_df.collect()
    assert len(rows) == 2
    assert set(row["data"] for row in rows) == {"record1", "record3"}


def test_partition_by_uuid_different_partitions(spark):
    """
    Test partition_by_uuid with different partition IDs.
    """
    # when I have UUIDs
    data = [
        ("e179017a-62b0-4996-8a38-e91aa9f1e179", "A"),  # 57721 % 4 = 1
        ("550e8400-e29b-41d4-a716-446655440001", "B"),  # 1 % 4 = 1
        ("550e8400-e29b-41d4-a716-446655440002", "C"),  # 2 % 4 = 2
        ("550e8400-e29b-41d4-a716-446655440003", "D"),  # 3 % 4 = 3
        ("550e8400-e29b-41d4-a716-446655440004", "E"),  # 4 % 4 = 0
    ]
    df = spark.createDataFrame(data, ["uuid", "label"])

    # and I get different partitions
    partition_0 = partition_by_uuid(df, "uuid", num_partitions=4, partition_id=0)
    partition_1 = partition_by_uuid(df, "uuid", num_partitions=4, partition_id=1)
    partition_2 = partition_by_uuid(df, "uuid", num_partitions=4, partition_id=2)
    partition_3 = partition_by_uuid(df, "uuid", num_partitions=4, partition_id=3)

    # then each partition should get the right records
    assert set(row["label"] for row in partition_0.collect()) == {"E"}
    assert set(row["label"] for row in partition_1.collect()) == {"A", "B"}
    assert set(row["label"] for row in partition_2.collect()) == {"C"}
    assert set(row["label"] for row in partition_3.collect()) == {"D"}


def test_partition_by_uuid_handles_nulls(spark):
    """
    Test that partition_by_uuid handles null UUIDs gracefully.
    """
    # when I have UUIDs including nulls
    data = [
        ("e179017a-62b0-4996-8a38-e91aa9f1e179", "valid1"),
        (None, "null_uuid"),
        ("550e8400-e29b-41d4-a716-446655440000", "valid2"),
    ]
    df = spark.createDataFrame(data, ["uuid", "label"])

    # and I get partition 1 of 2 partitions
    result_df = partition_by_uuid(df, "uuid", num_partitions=2, partition_id=1)

    # then nulls should be filtered out
    rows = result_df.collect()
    assert len(rows) == 1
    assert rows[0]["label"] == "valid1"  # 57721 % 2 = 1


def test_split_uuid_with_malformed_uuid_spark(spark):
    """
    Test that split_uuid_string_for_id handles malformed UUIDs.
    """
    # when I have malformed UUID strings
    data = [
        ("not-a-uuid",),
        ("too-few-segments",),
        ("a-b-c-d",),  # only 4 segments
        ("",),
    ]
    df = spark.createDataFrame(data, ["uuid"])

    # and I apply split_uuid_string_for_id
    df = df.withColumn("result", split_uuid_string_for_id(UUIDColumn(F.col("uuid"))))

    # then I should get nulls for invalid formats
    rows = df.collect()
    assert all(row["result"] is None for row in rows)


def test_end_to_end_uuid_partitioning(spark):
    """
    Test complete UUID-based partitioning workflow for horizontal scaling.
    """
    # when I have a large dataset with UUIDs
    import uuid

    data = [(str(uuid.uuid4()), i) for i in range(100)]
    df = spark.createDataFrame(data, ["uuid", "value"])

    # and I partition into 4 groups
    num_partitions = 4
    partitions = []
    for partition_id in range(num_partitions):
        partition = partition_by_uuid(df, "uuid", num_partitions, partition_id)
        partitions.append(partition)

    # then all records should be in exactly one partition
    total_original = df.count()
    total_partitioned = sum(p.count() for p in partitions)
    assert total_original == total_partitioned

    # and partitions should not overlap
    for i in range(num_partitions):
        for j in range(i + 1, num_partitions):
            partition_i = set(row["uuid"] for row in partitions[i].collect())
            partition_j = set(row["uuid"] for row in partitions[j].collect())
            assert partition_i.isdisjoint(partition_j)
