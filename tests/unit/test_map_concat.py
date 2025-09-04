import pyspark.sql.functions as F
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, MapType, StringType

from pyspark_toolkit.helpers import map_concat


@pytest.fixture
def spark():
    return SparkSession.builder.appName("test_map_concat").getOrCreate()


@pytest.fixture
def spark_with_last_win():
    """Spark session configured with LAST_WIN map key dedup policy."""
    spark = SparkSession.builder.appName("test_map_concat").getOrCreate()
    original_policy = spark.conf.get("spark.sql.mapKeyDedupPolicy", "EXCEPTION")
    spark.conf.set("spark.sql.mapKeyDedupPolicy", "LAST_WIN")
    yield spark
    spark.conf.set("spark.sql.mapKeyDedupPolicy", original_policy)


def test_when_two_maps_with_no_overlap_then_concatenates_all_keys(spark):
    """
    Test that map_concat properly merges maps with no duplicate keys.
    This ensures basic concatenation works correctly.
    """
    # when I have two maps with different keys
    df = spark.createDataFrame(
        [
            ({"a": 1, "b": 2}, {"c": 3, "d": 4}),
        ],
        ["map1", "map2"],
    )

    # and I concatenate them
    result_df = df.select(map_concat(F.col("map1"), F.col("map2")).alias("result"))
    result = result_df.collect()[0]["result"]

    # then I should get all keys from both maps
    expected = {"a": 1, "b": 2, "c": 3, "d": 4}
    assert result == expected


def test_when_maps_have_duplicate_keys_then_rightmost_wins(spark):
    """
    Test that map_concat uses right-override strategy for duplicate keys.
    This ensures the merge strategy works as expected.
    """
    # when I have maps with overlapping keys
    df = spark.createDataFrame(
        [
            ({"a": 1, "b": 2}, {"b": 20, "c": 3}),
        ],
        ["map1", "map2"],
    )

    # and I concatenate them
    result_df = df.select(map_concat(F.col("map1"), F.col("map2")).alias("result"))
    result = result_df.collect()[0]["result"]

    # then rightmost values should win for duplicate keys
    expected = {"a": 1, "b": 20, "c": 3}  # b=20 from map2, not b=2 from map1
    assert result == expected


def test_when_three_maps_with_overlaps_then_rightmost_wins(spark):
    """
    Test that map_concat works with multiple maps and rightmost precedence.
    This ensures the function handles variable arguments correctly.
    """
    # when I have three maps with overlapping keys
    df = spark.createDataFrame(
        [
            ({"a": 1, "b": 2}, {"b": 20, "c": 3}, {"a": 100, "d": 4}),
        ],
        ["map1", "map2", "map3"],
    )

    # and I concatenate all three
    result_df = df.select(map_concat(F.col("map1"), F.col("map2"), F.col("map3")).alias("result"))
    result = result_df.collect()[0]["result"]

    # then rightmost values should win
    expected = {"a": 100, "b": 20, "c": 3, "d": 4}  # a=100 from map3, b=20 from map2
    assert result == expected


def test_when_single_map_provided_then_returns_unchanged(spark):
    """
    Test that map_concat with single argument returns the map unchanged.
    This ensures edge case handling for single arguments.
    """
    # when I have a single map
    df = spark.createDataFrame(
        [
            ({"a": 1, "b": 2},),
        ],
        ["map1"],
    )

    # and I call map_concat with just one argument
    result_df = df.select(map_concat(F.col("map1")).alias("result"))
    result = result_df.collect()[0]["result"]

    # then I should get the original map unchanged
    expected = {"a": 1, "b": 2}
    assert result == expected


def test_when_empty_maps_provided_then_handles_correctly(spark):
    """
    Test that map_concat handles empty maps correctly.
    This ensures robust handling of edge cases.
    """
    # when I have empty maps and non-empty maps
    df = spark.createDataFrame(
        [
            ({}, {"a": 1}),
            ({"b": 2}, {}),
        ],
        ["map1", "map2"],
    )

    # and I concatenate them
    result_df = df.select(map_concat(F.col("map1"), F.col("map2")).alias("result"))
    results = [row["result"] for row in result_df.collect()]

    # then empty maps should not affect the result
    assert results[0] == {"a": 1}  # empty map1, map2 has content
    assert results[1] == {"b": 2}  # map1 has content, empty map2


def test_when_null_values_in_maps_then_preserves_nulls(spark):
    """
    Test that map_concat preserves null values in map entries.
    This ensures null handling works correctly.
    """
    # when I have maps with null values
    df = spark.createDataFrame(
        [
            ({"a": 1, "b": None}, {"b": 2, "c": None}),
        ],
        ["map1", "map2"],
    )

    # and I concatenate them
    result_df = df.select(map_concat(F.col("map1"), F.col("map2")).alias("result"))
    result = result_df.collect()[0]["result"]

    # then null values should be preserved and right map should win
    expected = {"a": 1, "b": 2, "c": None}  # b=2 from map2, c=None preserved
    assert result == expected


def test_when_string_keys_and_values_then_works_correctly(spark):
    """
    Test that map_concat works with string keys and values.
    This ensures the function works with different data types.
    """
    # when I have maps with string keys and values
    df = spark.createDataFrame(
        [
            ({"key1": "val1", "key2": "val2"}, {"key2": "val2_new", "key3": "val3"}),
        ],
        ["map1", "map2"],
    )

    # and I concatenate them
    result_df = df.select(map_concat(F.col("map1"), F.col("map2")).alias("result"))
    result = result_df.collect()[0]["result"]

    # then string concatenation should work with right override
    expected = {"key1": "val1", "key2": "val2_new", "key3": "val3"}
    assert result == expected


def test_doc_example_basic_usage(spark):
    """
    Test Example 1 from PySpark docs: Basic usage of map_concat.
    This ensures our function matches the documented behavior.
    """
    # when I have maps like in the documentation example
    df = spark.sql("SELECT map(1, 'a', 2, 'b') as map1, map(3, 'c') as map2")

    # and I concatenate them using our function
    result_df = df.select(map_concat(F.col("map1"), F.col("map2")).alias("result"))
    result = result_df.collect()[0]["result"]

    # then I should get the expected concatenated map
    expected = {1: "a", 2: "b", 3: "c"}
    assert result == expected


def test_doc_example_overlapping_keys(spark):
    """
    Test Example 2 from PySpark docs: map_concat with overlapping keys (LAST_WIN behavior).
    This ensures our function handles overlapping keys correctly.
    """
    # when I have maps with overlapping keys like in the documentation
    df = spark.sql("SELECT map(1, 'a', 2, 'b') as map1, map(2, 'c', 3, 'd') as map2")

    # and I concatenate them using our function
    result_df = df.select(map_concat(F.col("map1"), F.col("map2")).alias("result"))
    result = result_df.collect()[0]["result"]

    # then the rightmost map should win for duplicate keys
    expected = {1: "a", 2: "c", 3: "d"}  # key 2 has value 'c' from map2, not 'b' from map1
    assert result == expected


def test_doc_example_three_maps(spark):
    """
    Test Example 3 from PySpark docs: map_concat with three maps.
    This ensures our function works with multiple maps as documented.
    """
    # when I have three maps like in the documentation
    df = spark.sql("SELECT map(1, 'a') as map1, map(2, 'b') as map2, map(3, 'c') as map3")

    # and I concatenate all three using our function
    result_df = df.select(map_concat(F.col("map1"), F.col("map2"), F.col("map3")).alias("result"))
    result = result_df.collect()[0]["result"]

    # then I should get all keys from all maps
    expected = {1: "a", 2: "b", 3: "c"}
    assert result == expected


def test_doc_example_empty_map(spark):
    """
    Test Example 4 from PySpark docs: map_concat with empty map.
    This ensures our function handles empty maps as documented.
    """
    # when I have a regular map and an empty map like in the documentation
    df = spark.sql("SELECT map(1, 'a', 2, 'b') as map1, map() as map2")

    # and I concatenate them using our function
    result_df = df.select(map_concat(F.col("map1"), F.col("map2")).alias("result"))
    result = result_df.collect()[0]["result"]

    # then the empty map should not affect the result
    expected = {1: "a", 2: "b"}
    assert result == expected


def test_doc_example_null_values(spark):
    """
    Test Example 5 from PySpark docs: map_concat with null values.
    This ensures our function preserves null values as documented.
    """
    # when I have maps with null values like in the documentation
    df = spark.sql("SELECT map(1, 'a', 2, 'b') as map1, map(3, null) as map2")

    # and I concatenate them using our function
    result_df = df.select(map_concat(F.col("map1"), F.col("map2")).alias("result"))
    result = result_df.collect()[0]["result"]

    # then null values should be preserved in the result
    expected = {1: "a", 2: "b", 3: None}
    assert result == expected


def test_comparison_with_builtin_map_concat_last_win(spark_with_last_win):
    """
    Test that our map_concat produces identical results to built-in map_concat with LAST_WIN.
    This ensures our implementation matches PySpark's behavior exactly.
    """
    test_cases = [
        # Basic non-overlapping case
        ("SELECT map(1, 'a', 2, 'b') as map1, map(3, 'c') as map2", 2),
        # Overlapping keys case
        ("SELECT map(1, 'a', 2, 'b') as map1, map(2, 'c', 3, 'd') as map2", 2),
        # Three maps case
        ("SELECT map(1, 'a') as map1, map(2, 'b') as map2, map(3, 'c') as map3", 3),
        # Empty map case
        ("SELECT map(1, 'a', 2, 'b') as map1, map() as map2", 2),
        # Null values case
        ("SELECT map(1, 'a', 2, 'b') as map1, map(3, null) as map2", 2),
        # Complex overlapping case with multiple duplicates
        (
            "SELECT map(1, 'first', 2, 'second') as map1, map(2, 'override', 3, 'third') as map2, map(1, 'final') as map3",
            3,
        ),
    ]

    for sql_query, num_maps in test_cases:
        # when I have test data
        df = spark_with_last_win.sql(sql_query)

        # and I use built-in map_concat with LAST_WIN
        if num_maps == 2:
            builtin_result_df = df.select(F.map_concat("map1", "map2").alias("builtin_result"))
            custom_result_df = df.select(map_concat(F.col("map1"), F.col("map2")).alias("custom_result"))
        elif num_maps == 3:
            builtin_result_df = df.select(F.map_concat("map1", "map2", "map3").alias("builtin_result"))
            custom_result_df = df.select(map_concat(F.col("map1"), F.col("map2"), F.col("map3")).alias("custom_result"))

        builtin_result = builtin_result_df.collect()[0]["builtin_result"]
        custom_result = custom_result_df.collect()[0]["custom_result"]

        # then our function should produce identical results
        assert custom_result == builtin_result, (
            f"Mismatch for query: {sql_query}\nBuilt-in: {builtin_result}\nCustom: {custom_result}"
        )
