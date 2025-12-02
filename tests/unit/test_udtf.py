import pytest
from pyspark.sql import Row
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

try:
    from pyspark_toolkit.udtf import _as_dict, fdtf
except ImportError:
    pytest.skip("spark4_only not available", allow_module_level=True)


# Core Functionality Tests


@pytest.mark.spark40_only
def test_fdtf_with_no_arguments_appends_columns(spark):
    """
    Test that fdtf appends new columns to DataFrame without extra arguments.
    This ensures basic fdtf functionality works correctly.
    """
    # when I have a simple DataFrame
    df = spark.createDataFrame(
        [(1, "a"), (2, "b")],
        ["id", "value"],
    )

    # and I have an fdtf function that adds a new column
    @fdtf(output_schema=StructType([StructField("doubled", IntegerType())]))
    def add_doubled(row):
        yield (row["id"] * 2,)

    # and I apply the function
    result_df = add_doubled(df)
    results = result_df.collect()

    # then I should get original columns plus new column
    assert len(results) == 2
    assert results[0]["id"] == 1
    assert results[0]["value"] == "a"
    assert results[0]["doubled"] == 2
    assert results[1]["id"] == 2
    assert results[1]["value"] == "b"
    assert results[1]["doubled"] == 4


@pytest.mark.spark40_only
def test_fdtf_preserves_all_input_columns_and_values(spark):
    """
    Test that fdtf preserves all input columns and their values.
    This ensures data integrity is maintained.
    """
    # when I have a DataFrame with multiple columns
    df = spark.createDataFrame(
        [(1, "a", 10.5), (2, "b", 20.3)],
        ["id", "name", "score"],
    )

    # and I have an fdtf function that adds a new column
    @fdtf(output_schema=StructType([StructField("flag", StringType())]))
    def add_flag(row):
        yield ("ok",)

    # and I apply the function
    result_df = add_flag(df)
    results = result_df.collect()

    # then all original columns should be preserved with correct values
    assert results[0]["id"] == 1
    assert results[0]["name"] == "a"
    assert results[0]["score"] == 10.5
    assert results[0]["flag"] == "ok"
    assert results[1]["id"] == 2
    assert results[1]["name"] == "b"
    assert results[1]["score"] == 20.3
    assert results[1]["flag"] == "ok"


@pytest.mark.spark40_only
def test_fdtf_handles_multiple_rows_independently(spark):
    """
    Test that fdtf processes each row independently.
    This ensures row independence is maintained.
    """
    # when I have a DataFrame with multiple rows
    df = spark.createDataFrame(
        [(1, 10), (2, 20), (3, 30)],
        ["id", "value"],
    )

    # and I have an fdtf function that transforms each row
    @fdtf(output_schema=StructType([StructField("computed", IntegerType())]))
    def compute(row):
        yield (row["id"] + row["value"],)

    # and I apply the function
    result_df = compute(df)
    results = result_df.collect()

    # then each row should be processed independently with correct output
    assert len(results) == 3
    assert results[0]["computed"] == 11
    assert results[1]["computed"] == 22
    assert results[2]["computed"] == 33


@pytest.mark.spark40_only
def test_fdtf_with_positional_args_passes_values_correctly(spark):
    """
    Test that fdtf correctly passes positional arguments to the function.
    This ensures positional argument handling works.
    """
    # when I have a DataFrame
    df = spark.createDataFrame(
        [(1,), (2,)],
        ["id"],
    )

    # and I have an fdtf function that uses positional args
    @fdtf(output_schema=StructType([StructField("result", IntegerType())]))
    def add_args(row, arg1, arg2):
        yield (row["id"] + arg1 + arg2,)

    # and I call it with positional arguments
    result_df = add_args(df, 10, 20)
    results = result_df.collect()

    # then arguments should be passed correctly
    assert results[0]["result"] == 31  # 1 + 10 + 20
    assert results[1]["result"] == 32  # 2 + 10 + 20


@pytest.mark.spark40_only
def test_fdtf_with_keyword_args_passes_values_correctly(spark):
    """
    Test that fdtf correctly passes keyword arguments to the function.
    This ensures keyword argument handling works.
    """
    # when I have a DataFrame
    df = spark.createDataFrame(
        [(1,), (2,)],
        ["id"],
    )

    # and I have an fdtf function that uses kwargs
    @fdtf(output_schema=StructType([StructField("result", IntegerType())]))
    def add_kwargs(row, multiplier=1, offset=0):
        yield (row["id"] * multiplier + offset,)

    # and I call it with keyword arguments
    result_df = add_kwargs(df, multiplier=5, offset=10)
    results = result_df.collect()

    # then kwargs should be passed correctly
    assert results[0]["result"] == 15  # 1 * 5 + 10
    assert results[1]["result"] == 20  # 2 * 5 + 10


@pytest.mark.spark40_only
def test_fdtf_with_mixed_args_and_kwargs(spark):
    """
    Test that fdtf correctly handles both positional and keyword arguments.
    This ensures mixed argument handling works.
    """
    # when I have a DataFrame
    df = spark.createDataFrame(
        [(1,), (2,)],
        ["id"],
    )

    # and I have an fdtf function that uses both args and kwargs
    @fdtf(output_schema=StructType([StructField("result", IntegerType())]))
    def compute(row, base, multiplier=2, offset=0):
        yield (row["id"] * multiplier + base + offset,)

    # and I call it with mixed arguments
    result_df = compute(df, 100, multiplier=3, offset=5)
    results = result_df.collect()

    # then both positional and keyword args should be passed correctly
    assert results[0]["result"] == 108  # 1 * 3 + 100 + 5
    assert results[1]["result"] == 111  # 2 * 3 + 100 + 5


@pytest.mark.spark40_only
def test_fdtf_with_no_extra_args_only_receives_row(spark):
    """
    Test that fdtf works correctly when only the row parameter is used.
    This ensures the function works without extra arguments.
    """
    # when I have a DataFrame
    df = spark.createDataFrame(
        [(1, "a"), (2, "b")],
        ["id", "name"],
    )

    # and I have an fdtf function that only uses row
    @fdtf(output_schema=StructType([StructField("upper_name", StringType())]))
    def process_row(row):
        yield (row["name"].upper(),)

    # and I call it without extra arguments
    result_df = process_row(df)
    results = result_df.collect()

    # then function should work correctly with just row parameter
    assert results[0]["upper_name"] == "A"
    assert results[1]["upper_name"] == "B"


# Schema and Output Tests


@pytest.mark.spark40_only
def test_fdtf_output_schema_matches_specification(spark):
    """
    Test that fdtf produces output with correct schema.
    This ensures schema handling is correct.
    """
    # when I have a DataFrame
    df = spark.createDataFrame(
        [(1, "a")],
        ["id", "value"],
    )

    # and I have an fdtf with specific output schema
    output_schema = StructType(
        [
            StructField("new_int", IntegerType()),
            StructField("new_str", StringType()),
        ]
    )

    @fdtf(output_schema=output_schema)
    def add_columns(row):
        yield (42, "test")

    # and I apply the function
    result_df = add_columns(df)

    # then output schema should match base schema + output schema
    expected_fields = ["id", "value", "new_int", "new_str"]
    actual_fields = [f.name for f in result_df.schema.fields]
    assert actual_fields == expected_fields

    # and types should be correct
    assert result_df.schema["new_int"].dataType == IntegerType()
    assert result_df.schema["new_str"].dataType == StringType()


@pytest.mark.spark40_only
def test_fdtf_function_yielding_multiple_rows_per_input(spark):
    """
    Test that fdtf handles functions that yield multiple output rows per input.
    This ensures the exploding behavior works correctly.
    """
    # when I have a DataFrame with single row
    df = spark.createDataFrame(
        [(1, "a")],
        ["id", "value"],
    )

    # and I have an fdtf function that yields multiple rows
    @fdtf(output_schema=StructType([StructField("iteration", IntegerType())]))
    def explode_rows(row):
        for i in range(3):
            yield (i,)

    # and I apply the function
    result_df = explode_rows(df)
    results = result_df.collect()

    # then I should get multiple output rows per input row
    assert len(results) == 3
    assert results[0]["id"] == 1
    assert results[0]["value"] == "a"
    assert results[0]["iteration"] == 0
    assert results[1]["iteration"] == 1
    assert results[2]["iteration"] == 2


# Edge Case Tests


@pytest.mark.spark40_only
def test_fdtf_with_empty_dataframe(spark):
    """
    Test that fdtf handles empty DataFrames correctly.
    This ensures edge case handling for empty input.
    """
    # when I have an empty DataFrame with defined schema
    schema = StructType(
        [
            StructField("id", IntegerType()),
            StructField("value", StringType()),
        ]
    )
    df = spark.createDataFrame([], schema)

    # and I have an fdtf function
    @fdtf(output_schema=StructType([StructField("result", IntegerType())]))
    def process(row):
        yield (42,)

    # and I apply the function
    result_df = process(df)
    results = result_df.collect()

    # then I should get empty result with correct schema
    assert len(results) == 0
    expected_fields = ["id", "value", "result"]
    actual_fields = [f.name for f in result_df.schema.fields]
    assert actual_fields == expected_fields


@pytest.mark.spark40_only
def test_fdtf_with_null_values_in_input(spark):
    """
    Test that fdtf handles null values in input DataFrame correctly.
    This ensures null handling works properly.
    """
    # when I have a DataFrame with null values
    df = spark.createDataFrame(
        [(1, None), (None, "b"), (3, "c")],
        ["id", "value"],
    )

    # and I have an fdtf function that handles nulls
    @fdtf(output_schema=StructType([StructField("has_null", StringType())]))
    def check_nulls(row):
        has_null = "yes" if row["id"] is None or row["value"] is None else "no"
        yield (has_null,)

    # and I apply the function
    result_df = check_nulls(df)
    results = result_df.collect()

    # then nulls should be handled correctly and preserved
    assert results[0]["id"] == 1
    assert results[0]["value"] is None
    assert results[0]["has_null"] == "yes"
    assert results[1]["id"] is None
    assert results[1]["value"] == "b"
    assert results[1]["has_null"] == "yes"
    assert results[2]["id"] == 3
    assert results[2]["value"] == "c"
    assert results[2]["has_null"] == "no"


# Helper Function Tests


@pytest.mark.spark40_only
def test_as_dict_with_pyspark_row(spark):
    """
    Test that _as_dict converts PySpark Row to dict recursively.
    This ensures Row conversion works correctly.
    """
    # when I have a PySpark Row
    row = Row(id=1, name="test", nested=Row(a=10, b=20))

    # and I convert it to dict
    result = _as_dict(row)

    # then I should get a recursive dict representation
    assert result["id"] == 1
    assert result["name"] == "test"
    assert isinstance(result["nested"], dict)
    assert result["nested"]["a"] == 10
    assert result["nested"]["b"] == 20


@pytest.mark.spark40_only
def test_as_dict_with_dict_like_object(spark):
    """
    Test that _as_dict handles dict-like objects without asDict method.
    This ensures compatibility with dict-like objects.
    """
    # when I have a dict-like object (regular dict)
    obj = {"id": 1, "name": "test"}

    # and I convert it to dict
    result = _as_dict(obj)

    # then I should get a dict
    assert isinstance(result, dict)
    assert result["id"] == 1
    assert result["name"] == "test"
