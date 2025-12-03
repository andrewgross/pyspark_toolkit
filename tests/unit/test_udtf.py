import pytest
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

try:
    from pyspark_toolkit.udtf import _validate_fdtf_signature, fdtf
except ImportError:
    pytest.skip("spark40_only not available", allow_module_level=True)


# Helper functions for metadata assertions


def assert_success(row, expected_attempts=1):
    """Assert a row succeeded and has expected number of attempts."""
    metadata = row["_metadata"]
    assert len(metadata) == expected_attempts
    # Last attempt should have no error
    assert metadata[-1]["error"] is None


def assert_failure(row, expected_attempts=1):
    """Assert a row failed and has expected number of attempts."""
    metadata = row["_metadata"]
    assert len(metadata) == expected_attempts
    # All attempts should have errors
    for attempt in metadata:
        assert attempt["error"] is not None


def get_last_error(row):
    """Get the error message from the last attempt."""
    metadata = row["_metadata"]
    return metadata[-1]["error"]


# fdtf (Flexible DataFrame Table Function) Tests


@pytest.mark.spark40_only
def test_fdtf_basic_concurrent_execution(spark):
    """
    Test that fdtf processes rows concurrently and returns correct results.
    This ensures the basic concurrent execution pattern works.
    """
    # when I have a DataFrame
    df = spark.createDataFrame(
        [(1, "a"), (2, "b"), (3, "c")],
        ["id", "value"],
    )

    # and I have a fdtf function with simple init/cleanup
    def my_init(self):
        self.counter = 0

    def my_cleanup(self):  # noqa: ARG001
        pass

    @fdtf(
        output_schema="doubled INT",
        init_fn=my_init,
        cleanup_fn=my_cleanup,
        max_workers=2,
    )
    def double_id(self, row):  # noqa: ARG001
        return (row["id"] * 2,)

    # and I apply the function
    result_df = double_id(df)
    results = sorted(result_df.collect(), key=lambda r: r["id"])

    # then I should get original columns plus new column plus metadata
    assert len(results) == 3
    assert results[0]["id"] == 1
    assert results[0]["value"] == "a"
    assert results[0]["doubled"] == 2
    assert_success(results[0])

    assert results[1]["doubled"] == 4
    assert results[2]["doubled"] == 6


@pytest.mark.spark40_only
def test_fdtf_without_init_fn(spark):
    """
    Test that fdtf works without providing init_fn.
    This ensures the simple usage pattern works.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,), (2,), (3,)], ["id"])

    # and I have a fdtf function without init_fn
    @fdtf(output_schema="doubled INT", max_workers=2)
    def double_id(self, row):  # noqa: ARG001
        return (row["id"] * 2,)

    # and I apply the function
    result_df = double_id(df)
    results = sorted(result_df.collect(), key=lambda r: r["id"])

    # then it should work correctly
    assert len(results) == 3
    assert results[0]["doubled"] == 2
    assert results[1]["doubled"] == 4
    assert results[2]["doubled"] == 6


@pytest.mark.spark40_only
def test_fdtf_metadata_contains_execution_details(spark):
    """
    Test that fdtf _metadata column contains proper execution details.
    This ensures metadata is captured correctly as an array of attempt records.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,)], ["id"])

    # and I have a fdtf function
    @fdtf(output_schema="result INT", max_workers=1)
    def process(self, row):  # noqa: ARG001
        return (42,)

    # and I apply the function
    result_df = process(df)
    result = result_df.collect()[0]

    # then metadata should be an array with one attempt record
    metadata = result["_metadata"]
    assert len(metadata) == 1

    # and the attempt record should have expected fields
    attempt = metadata[0]
    assert attempt["attempt"] == 1
    assert attempt["started_at"] is not None
    assert attempt["duration_ms"] is not None
    assert attempt["error"] is None
    assert isinstance(attempt["duration_ms"], int)


@pytest.mark.spark40_only
def test_fdtf_preserves_all_input_columns(spark):
    """
    Test that fdtf preserves all input columns in the output.
    This ensures data integrity is maintained.
    """
    # when I have a DataFrame with multiple columns
    df = spark.createDataFrame(
        [(1, "a", 10.5), (2, "b", 20.3)],
        ["id", "name", "score"],
    )

    # and I have a fdtf function
    @fdtf(output_schema="flag STRING", max_workers=2)
    def add_flag(self, row):  # noqa: ARG001
        return ("ok",)

    # and I apply the function
    result_df = add_flag(df)
    results = sorted(result_df.collect(), key=lambda r: r["id"])

    # then all original columns should be preserved
    assert results[0]["id"] == 1
    assert results[0]["name"] == "a"
    assert results[0]["score"] == 10.5
    assert results[0]["flag"] == "ok"


@pytest.mark.spark40_only
def test_fdtf_with_positional_args(spark):
    """
    Test that fdtf correctly passes positional arguments to the function.
    This ensures argument handling works.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,), (2,)], ["id"])

    # and I have a fdtf function that uses positional args
    @fdtf(output_schema="result INT", max_workers=2)
    def add_args(self, row, arg1, arg2):  # noqa: ARG001
        return (row["id"] + arg1 + arg2,)

    # and I call it with positional arguments
    result_df = add_args(df, 10, 20)
    results = sorted(result_df.collect(), key=lambda r: r["id"])

    # then arguments should be passed correctly
    assert results[0]["result"] == 31  # 1 + 10 + 20
    assert results[1]["result"] == 32  # 2 + 10 + 20


@pytest.mark.spark40_only
def test_fdtf_with_keyword_args(spark):
    """
    Test that fdtf correctly passes keyword arguments to the function.
    This ensures kwarg handling works.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,), (2,)], ["id"])

    # and I have a fdtf function that uses kwargs
    @fdtf(output_schema="result INT", max_workers=2)
    def compute(self, row, multiplier=1, offset=0):  # noqa: ARG001
        return (row["id"] * multiplier + offset,)

    # and I call it with keyword arguments
    result_df = compute(df, multiplier=5, offset=10)
    results = sorted(result_df.collect(), key=lambda r: r["id"])

    # then kwargs should be passed correctly
    assert results[0]["result"] == 15  # 1 * 5 + 10
    assert results[1]["result"] == 20  # 2 * 5 + 10


@pytest.mark.spark40_only
def test_fdtf_error_handling_captures_exceptions(spark):
    """
    Test that fdtf captures exceptions in the metadata.
    This ensures error handling works correctly.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,), (2,)], ["id"])

    # and I have a fdtf function that raises an error for some rows
    @fdtf(output_schema="result INT", max_workers=2)
    def may_fail(self, row):  # noqa: ARG001
        if row["id"] == 1:
            raise ValueError("Simulated error")
        return (row["id"] * 10,)

    # and I apply the function
    result_df = may_fail(df)
    results = sorted(result_df.collect(), key=lambda r: r["id"])

    # then the error row should have result null and error in metadata
    assert results[0]["id"] == 1
    assert results[0]["result"] is None
    assert_failure(results[0], expected_attempts=1)
    error = get_last_error(results[0])
    assert "ValueError" in error
    assert "Simulated error" in error

    # and the success row should have result and no error
    assert results[1]["id"] == 2
    assert results[1]["result"] == 20
    assert_success(results[1])


@pytest.mark.spark40_only
def test_fdtf_with_max_retries_retries_on_failure(spark):
    """
    Test that fdtf retries when max_retries is set.
    This ensures retry logic works correctly.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,)], ["id"])

    # and I have a fdtf function that fails twice then succeeds
    def my_init(self):
        self.attempt_count = 0

    @fdtf(
        output_schema="result INT",
        init_fn=my_init,
        max_workers=1,
        max_retries=3,
    )
    def flaky_fn(self, row):  # noqa: ARG001
        self.attempt_count += 1
        if self.attempt_count < 3:
            raise Exception("temporary failure")
        return (42,)

    # and I apply the function
    result_df = flaky_fn(df)
    result = result_df.collect()[0]

    # then it should eventually succeed
    assert result["result"] == 42

    # and metadata should show all 3 attempts (2 failures + 1 success)
    metadata = result["_metadata"]
    assert len(metadata) == 3
    assert metadata[0]["error"] is not None  # first attempt failed
    assert metadata[1]["error"] is not None  # second attempt failed
    assert metadata[2]["error"] is None  # third attempt succeeded


@pytest.mark.spark40_only
def test_fdtf_with_max_retries_fails_after_exhausting_retries(spark):
    """
    Test that fdtf fails after exhausting max retries.
    This ensures retry limits are respected.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,)], ["id"])

    # and I have a fdtf function that always fails
    @fdtf(
        output_schema="result INT",
        max_workers=1,
        max_retries=2,
    )
    def always_fails(self, row):  # noqa: ARG001
        raise Exception("Service Unavailable")

    # and I apply the function
    result_df = always_fails(df)
    result = result_df.collect()[0]

    # then the result should be null
    assert result["result"] is None

    # and metadata should show all 3 attempts failed (initial + 2 retries)
    metadata = result["_metadata"]
    assert len(metadata) == 3
    for attempt in metadata:
        assert attempt["error"] is not None
        assert "Service Unavailable" in attempt["error"]


@pytest.mark.spark40_only
def test_fdtf_no_retry_by_default(spark):
    """
    Test that fdtf does not retry when max_retries is 0 (default).
    This ensures default no-retry behavior.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,)], ["id"])

    # and I have a fdtf function that fails (using default max_retries=0)
    @fdtf(
        output_schema="result INT",
        max_workers=1,
    )
    def fails_immediately(self, row):  # noqa: ARG001
        raise ValueError("This is an error")

    # and I apply the function
    result_df = fails_immediately(df)
    result = result_df.collect()[0]

    # then it should fail without retrying (metadata shows only 1 attempt)
    assert_failure(result, expected_attempts=1)
    error = get_last_error(result)
    assert "ValueError" in error


@pytest.mark.spark40_only
def test_fdtf_resources_passed_to_function(spark):
    """
    Test that fdtf passes resources from init_fn to the processing function.
    This ensures resource injection works correctly.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,), (2,)], ["id"])

    # and I have init_fn that sets attributes on self
    def my_init(self):
        self.multiplier = 10
        self.prefix = "result_"

    # and I have a fdtf function that uses self attributes
    @fdtf(
        output_schema="computed STRING",
        init_fn=my_init,
        max_workers=2,
    )
    def use_resources(self, row) -> str:  # noqa: ARG001
        value = row["id"] * self.multiplier
        return f"{self.prefix}{value}"

    # and I apply the function
    result_df = use_resources(df)
    results = sorted(result_df.collect(), key=lambda r: r["id"])

    # then resources should be used correctly
    assert results[0]["computed"] == "result_10"
    assert results[1]["computed"] == "result_20"


@pytest.mark.spark40_only
def test_fdtf_with_ddl_string_schema(spark):
    """
    Test that fdtf works with DDL string output schema.
    This ensures DDL string schema support works.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,), (2,)], ["id"])

    # and I have a fdtf function with DDL string schema
    @fdtf(output_schema="doubled INT, label STRING", max_workers=2)
    def transform(self, row):  # noqa: ARG001
        return (row["id"] * 2, f"row_{row['id']}")

    # and I apply the function
    result_df = transform(df)
    results = sorted(result_df.collect(), key=lambda r: r["id"])

    # then output should match DDL schema
    assert results[0]["doubled"] == 2
    assert results[0]["label"] == "row_1"
    assert results[1]["doubled"] == 4
    assert results[1]["label"] == "row_2"


@pytest.mark.spark40_only
def test_fdtf_output_schema_structure(spark):
    """
    Test that fdtf produces correct output schema structure.
    This ensures schema ordering and types are correct.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1, "a")], ["id", "value"])

    # and I have a fdtf function
    @fdtf(
        output_schema=StructType(
            [
                StructField("result", IntegerType()),
                StructField("label", StringType()),
            ]
        ),
        max_workers=1,
    )
    def transform(self, row):  # noqa: ARG001
        return (42, "test")

    # and I apply the function
    result_df = transform(df)

    # then schema should be: input cols + user output cols + _metadata
    expected_fields = ["id", "value", "result", "label", "_metadata"]
    actual_fields = [f.name for f in result_df.schema.fields]
    assert actual_fields == expected_fields

    # and types should be correct
    assert result_df.schema["result"].dataType == IntegerType()
    assert result_df.schema["label"].dataType == StringType()

    # and metadata should be an array of attempt structs
    metadata_type = result_df.schema["_metadata"].dataType
    assert isinstance(metadata_type, ArrayType)

    # and each attempt struct should have the expected fields
    attempt_struct = metadata_type.elementType
    assert isinstance(attempt_struct, StructType)
    assert attempt_struct["attempt"].dataType == IntegerType()
    assert attempt_struct["started_at"].dataType == TimestampType()
    assert attempt_struct["duration_ms"].dataType == LongType()
    assert attempt_struct["error"].dataType == StringType()


@pytest.mark.spark40_only
def test_fdtf_with_empty_dataframe(spark):
    """
    Test that fdtf handles empty DataFrames correctly.
    This ensures edge case handling for empty input.
    """
    # when I have an empty DataFrame
    schema = StructType(
        [
            StructField("id", IntegerType()),
            StructField("value", StringType()),
        ]
    )
    df = spark.createDataFrame([], schema)

    # and I have a fdtf function
    @fdtf(output_schema="result INT", max_workers=2)
    def process(self, row):  # noqa: ARG001
        return (42,)

    # and I apply the function
    result_df = process(df)
    results = result_df.collect()

    # then I should get empty result with correct schema
    assert len(results) == 0
    expected_fields = ["id", "value", "result", "_metadata"]
    actual_fields = [f.name for f in result_df.schema.fields]
    assert actual_fields == expected_fields


@pytest.mark.spark40_only
def test_fdtf_with_max_retries_retries_any_exception(spark):
    """
    Test that fdtf retries any exception when max_retries is set.
    This ensures retry works for all exception types.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,)], ["id"])

    # and I have a fdtf function that raises an exception
    def my_init(self):
        self.attempt_count = 0

    @fdtf(
        output_schema="result INT",
        init_fn=my_init,
        max_workers=1,
        max_retries=2,
    )
    def retries_on_error(self, row):  # noqa: ARG001
        self.attempt_count += 1
        if self.attempt_count < 2:
            raise ConnectionError("Network failure")
        return (99,)

    # and I apply the function
    result_df = retries_on_error(df)
    result = result_df.collect()[0]

    # then it should succeed after retry
    assert result["result"] == 99

    # and metadata should show 2 attempts (1 failure + 1 success)
    metadata = result["_metadata"]
    assert len(metadata) == 2
    assert metadata[0]["error"] is not None  # first attempt failed
    assert metadata[1]["error"] is None  # second attempt succeeded


@pytest.mark.spark40_only
def test_fdtf_zero_retries_fails_immediately(spark):
    """
    Test that fdtf with max_retries=0 does not retry on failure.
    This ensures default no-retry behavior.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,)], ["id"])

    # and I have a fdtf function with max_retries=0 (the default)
    @fdtf(output_schema="result INT", max_workers=1, max_retries=0)
    def fails_once(self, row):  # noqa: ARG001
        raise Exception("Service Unavailable")

    # and I apply the function
    result_df = fails_once(df)
    result = result_df.collect()[0]

    # then it should fail without retrying (only 1 attempt)
    assert_failure(result, expected_attempts=1)


# Single Value Return Tests


@pytest.mark.spark40_only
def test_fdtf_with_single_value_return(spark):
    """
    Test that fdtf handles single value returns without explicit tuple wrapping.
    This ensures users can return a single value directly.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,), (2,), (3,)], ["id"])

    # and I have a fdtf function that returns a single value (not a tuple)
    @fdtf(output_schema="doubled INT", max_workers=2)
    def double_value(self, row):  # noqa: ARG001
        return row["id"] * 2  # returns int, not (int,)

    # and I apply the function
    result_df = double_value(df)
    results = sorted(result_df.collect(), key=lambda r: r["id"])

    # then it should work correctly
    assert len(results) == 3
    assert results[0]["doubled"] == 2
    assert results[1]["doubled"] == 4
    assert results[2]["doubled"] == 6


@pytest.mark.spark40_only
def test_fdtf_with_single_string_value_return(spark):
    """
    Test that fdtf handles single string value returns.
    This ensures string values are wrapped correctly.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1, "hello"), (2, "world")], ["id", "value"])

    # and I have a fdtf function that returns a single string value
    @fdtf(output_schema="upper STRING", max_workers=2)
    def upper_value(self, row):  # noqa: ARG001
        return row["value"].upper()  # returns str, not (str,)

    # and I apply the function
    result_df = upper_value(df)
    results = sorted(result_df.collect(), key=lambda r: r["id"])

    # then it should work correctly
    assert results[0]["upper"] == "HELLO"
    assert results[1]["upper"] == "WORLD"


@pytest.mark.spark40_only
def test_fdtf_with_tuple_return_still_works(spark):
    """
    Test that fdtf still works with explicit tuple returns.
    This ensures backwards compatibility with tuple returns.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,), (2,)], ["id"])

    # and I have a fdtf function that returns an explicit tuple
    @fdtf(output_schema="doubled INT", max_workers=2)
    def double_value(self, row):  # noqa: ARG001
        return (row["id"] * 2,)  # explicit tuple

    # and I apply the function
    result_df = double_value(df)
    results = sorted(result_df.collect(), key=lambda r: r["id"])

    # then it should work correctly
    assert results[0]["doubled"] == 2
    assert results[1]["doubled"] == 4


@pytest.mark.spark40_only
def test_fdtf_with_multi_column_tuple_return(spark):
    """
    Test that fdtf correctly handles multi-column tuple returns.
    This ensures tuple returns with multiple values work.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "value"])

    # and I have a fdtf function that returns multiple columns
    @fdtf(output_schema="doubled INT, upper STRING", max_workers=2)
    def transform(self, row):  # noqa: ARG001
        return (row["id"] * 2, row["value"].upper())

    # and I apply the function
    result_df = transform(df)
    results = sorted(result_df.collect(), key=lambda r: r["id"])

    # then both columns should be populated correctly
    assert results[0]["doubled"] == 2
    assert results[0]["upper"] == "A"
    assert results[1]["doubled"] == 4
    assert results[1]["upper"] == "B"


# Custom Column Names Tests


@pytest.mark.spark40_only
def test_fdtf_with_custom_metadata_column_name(spark):
    """
    Test that fdtf uses custom metadata column name when specified.
    This ensures the metadata_column parameter works correctly.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,)], ["id"])

    # and I have a fdtf function with custom metadata column name
    @fdtf(output_schema="result INT", max_workers=1, metadata_column="execution_info")
    def process(self, row):  # noqa: ARG001
        return (42,)

    # and I apply the function
    result_df = process(df)

    # then the metadata column should use the custom name
    field_names = [f.name for f in result_df.schema.fields]
    assert "execution_info" in field_names
    assert "_metadata" not in field_names

    # and the custom metadata column should be an array with attempt structs
    metadata_type = result_df.schema["execution_info"].dataType
    assert isinstance(metadata_type, ArrayType)
    attempt_struct = metadata_type.elementType
    assert isinstance(attempt_struct, StructType)
    assert "attempt" in [f.name for f in attempt_struct.fields]
    assert "started_at" in [f.name for f in attempt_struct.fields]
    assert "duration_ms" in [f.name for f in attempt_struct.fields]
    assert "error" in [f.name for f in attempt_struct.fields]


@pytest.mark.spark40_only
def test_fdtf_with_custom_metadata_column_has_correct_data(spark):
    """
    Test that fdtf with custom metadata column name contains correct data.
    This ensures custom column naming doesn't affect data integrity.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,)], ["id"])

    # and I have a fdtf function with custom metadata column that fails
    @fdtf(
        output_schema="result INT",
        max_workers=1,
        metadata_column="meta",
    )
    def fails(self, row):  # noqa: ARG001
        raise ValueError("Test error")

    # and I apply the function
    result_df = fails(df)
    result = result_df.collect()[0]

    # then metadata should be in the custom column with correct data
    metadata = result["meta"]
    assert len(metadata) == 1
    assert metadata[0]["attempt"] == 1
    assert metadata[0]["error"] is not None
    assert "ValueError" in metadata[0]["error"]


# Concurrency Tests


@pytest.mark.spark40_only
def test_fdtf_concurrent_execution_with_shared_client(spark):
    """
    Test that fdtf correctly shares resources across concurrent workers.
    This ensures the thread pool and init_fn work together properly.
    """
    import threading
    import time

    # when I have a DataFrame with enough rows to exercise concurrency
    df = spark.createDataFrame([(i,) for i in range(20)], ["id"])

    # and I have init_fn that creates a shared "client" with thread-safe counter
    def my_init(self):
        self.lock = threading.Lock()
        self.call_count = 0
        self.max_concurrent = 0
        self.current_concurrent = 0

    # and I have a fdtf function that simulates work and tracks concurrency
    @fdtf(
        output_schema="result INT",
        init_fn=my_init,
        max_workers=5,
    )
    def track_concurrency(self, row):
        with self.lock:
            self.call_count += 1
            self.current_concurrent += 1
            if self.current_concurrent > self.max_concurrent:
                self.max_concurrent = self.current_concurrent

        # simulate some work
        time.sleep(0.05)

        with self.lock:
            self.current_concurrent -= 1

        return (row["id"] * 2,)

    # and I apply the function
    result_df = track_concurrency(df)
    results = result_df.collect()

    # then all rows should be processed correctly
    assert len(results) == 20
    result_ids = {r["id"] for r in results}
    assert result_ids == set(range(20))

    # and all results should have correct doubled values and no errors
    for r in results:
        assert r["result"] == r["id"] * 2
        assert_success(r)


@pytest.mark.spark40_only
def test_fdtf_shared_http_client_simulation(spark):
    """
    Test that fdtf can share a simulated HTTP client across concurrent calls.
    This ensures realistic usage patterns with shared connections work.
    """
    import threading
    from collections import defaultdict

    # when I have a DataFrame with multiple rows
    df = spark.createDataFrame(
        [(1, "apple"), (2, "banana"), (3, "cherry"), (4, "date"), (5, "elderberry")],
        ["id", "fruit"],
    )

    # and I have a mock HTTP client that tracks requests
    class MockHttpClient:
        def __init__(self):
            self.request_count = 0
            self.lock = threading.Lock()
            self.responses = defaultdict(str)

        def get(self, path: str) -> str:
            with self.lock:
                self.request_count += 1
            # simulate network latency
            import time

            time.sleep(0.02)
            return f"response_for_{path}"

    def my_init(self):
        self.http = MockHttpClient()

    def my_cleanup(self):
        # in real code, this would close connections
        pass

    @fdtf(
        output_schema="api_response STRING",
        init_fn=my_init,
        cleanup_fn=my_cleanup,
        max_workers=3,
    )
    def call_api(self, row):
        response = self.http.get(row["fruit"])
        return (response,)

    # and I apply the function
    result_df = call_api(df)
    results = sorted(result_df.collect(), key=lambda r: r["id"])

    # then all rows should have responses
    assert len(results) == 5
    assert results[0]["api_response"] == "response_for_apple"
    assert results[1]["api_response"] == "response_for_banana"
    assert results[2]["api_response"] == "response_for_cherry"
    assert results[3]["api_response"] == "response_for_date"
    assert results[4]["api_response"] == "response_for_elderberry"

    # and no errors should have occurred
    for r in results:
        assert_success(r)


@pytest.mark.spark40_only
def test_fdtf_thread_safety_with_shared_state(spark):
    """
    Test that fdtf maintains thread safety when workers modify shared state.
    This ensures concurrent modifications don't cause race conditions.
    """
    import threading

    # when I have a DataFrame with rows that will be processed concurrently
    df = spark.createDataFrame([(i,) for i in range(50)], ["id"])

    # and I have init_fn with shared state protected by a lock
    def my_init(self):
        self.lock = threading.Lock()
        self.processed_ids = []

    @fdtf(
        output_schema="squared INT",
        init_fn=my_init,
        max_workers=10,
    )
    def thread_safe_process(self, row):
        row_id = row["id"]

        # safely append to shared list
        with self.lock:
            self.processed_ids.append(row_id)

        return (row_id * row_id,)

    # and I apply the function
    result_df = thread_safe_process(df)
    results = result_df.collect()

    # then all rows should be processed
    assert len(results) == 50

    # and all results should be correct
    for r in results:
        assert r["squared"] == r["id"] ** 2
        assert_success(r)


# Non-Concurrent (Sequential) Execution Tests


@pytest.mark.spark40_only
def test_fdtf_sequential_execution_with_max_workers_none(spark):
    """
    Test that fdtf processes rows sequentially when max_workers is None.
    This ensures the non-concurrent path works correctly.
    """
    # when I have a DataFrame
    df = spark.createDataFrame(
        [(1, "a"), (2, "b"), (3, "c")],
        ["id", "value"],
    )

    # and I have a fdtf function with max_workers=None
    @fdtf(output_schema="doubled INT", max_workers=None)
    def double_id(self, row):  # noqa: ARG001
        return (row["id"] * 2,)

    # and I apply the function
    result_df = double_id(df)
    results = sorted(result_df.collect(), key=lambda r: r["id"])

    # then I should get correct results
    assert len(results) == 3
    assert results[0]["doubled"] == 2
    assert results[1]["doubled"] == 4
    assert results[2]["doubled"] == 6

    # and metadata should be present
    for r in results:
        assert_success(r)


@pytest.mark.spark40_only
def test_fdtf_sequential_execution_with_max_workers_zero(spark):
    """
    Test that fdtf processes rows sequentially when max_workers is 0.
    This ensures 0 is treated the same as None.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,), (2,)], ["id"])

    # and I have a fdtf function with max_workers=0
    @fdtf(output_schema="squared INT", max_workers=0)
    def square_id(self, row):  # noqa: ARG001
        return (row["id"] ** 2,)

    # and I apply the function
    result_df = square_id(df)
    results = sorted(result_df.collect(), key=lambda r: r["id"])

    # then I should get correct results
    assert len(results) == 2
    assert results[0]["squared"] == 1
    assert results[1]["squared"] == 4


@pytest.mark.spark40_only
def test_fdtf_sequential_with_init_and_cleanup(spark):
    """
    Test that fdtf sequential mode still calls init_fn and cleanup_fn.
    This ensures resource management works without threading.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,), (2,)], ["id"])

    # and I have init/cleanup functions
    def my_init(self):
        self.multiplier = 10

    def my_cleanup(self):  # noqa: ARG001
        pass

    # and I have a fdtf function with max_workers=None
    @fdtf(
        output_schema="result INT",
        init_fn=my_init,
        cleanup_fn=my_cleanup,
        max_workers=None,
    )
    def multiply(self, row):
        return (row["id"] * self.multiplier,)

    # and I apply the function
    result_df = multiply(df)
    results = sorted(result_df.collect(), key=lambda r: r["id"])

    # then init_fn should have been called (multiplier should work)
    assert results[0]["result"] == 10
    assert results[1]["result"] == 20


@pytest.mark.spark40_only
def test_fdtf_sequential_with_retries(spark):
    """
    Test that fdtf sequential mode supports retries.
    This ensures retry logic works without threading.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,)], ["id"])

    # and I have a function that fails then succeeds
    def my_init(self):
        self.attempt_count = 0

    @fdtf(
        output_schema="result INT",
        init_fn=my_init,
        max_workers=None,
        max_retries=2,
    )
    def flaky_fn(self, row):  # noqa: ARG001
        self.attempt_count += 1
        if self.attempt_count < 2:
            raise Exception("temporary failure")
        return (42,)

    # and I apply the function
    result_df = flaky_fn(df)
    result = result_df.collect()[0]

    # then it should eventually succeed
    assert result["result"] == 42

    # and metadata should show 2 attempts
    metadata = result["_metadata"]
    assert len(metadata) == 2
    assert metadata[0]["error"] is not None
    assert metadata[1]["error"] is None


@pytest.mark.spark40_only
def test_fdtf_sequential_error_handling(spark):
    """
    Test that fdtf sequential mode captures errors correctly.
    This ensures error handling works without threading.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,), (2,)], ["id"])

    # and I have a function that fails for some rows
    @fdtf(output_schema="result INT", max_workers=None)
    def may_fail(self, row):  # noqa: ARG001
        if row["id"] == 1:
            raise ValueError("Simulated error")
        return (row["id"] * 10,)

    # and I apply the function
    result_df = may_fail(df)
    results = sorted(result_df.collect(), key=lambda r: r["id"])

    # then the error row should have null result and error in metadata
    assert results[0]["result"] is None
    assert_failure(results[0])
    error = get_last_error(results[0])
    assert "ValueError" in error

    # and the success row should have result and no error
    assert results[1]["result"] == 20
    assert_success(results[1])


@pytest.mark.spark40_only
def test_fdtf_sequential_preserves_row_order(spark):
    """
    Test that fdtf sequential mode preserves row order.
    Unlike concurrent mode, sequential should maintain input order.
    """
    # when I have a DataFrame with specific order
    df = spark.createDataFrame([(3,), (1,), (2,)], ["id"])

    # and I have a fdtf function with max_workers=None
    @fdtf(output_schema="doubled INT", max_workers=None)
    def double_id(self, row):  # noqa: ARG001
        return (row["id"] * 2,)

    # and I apply the function
    result_df = double_id(df)
    results = result_df.collect()

    # then results should be in original order (3, 1, 2)
    assert results[0]["id"] == 3
    assert results[0]["doubled"] == 6
    assert results[1]["id"] == 1
    assert results[1]["doubled"] == 2
    assert results[2]["id"] == 2
    assert results[2]["doubled"] == 4


@pytest.mark.spark40_only
def test_fdtf_sequential_with_args_and_kwargs(spark):
    """
    Test that fdtf sequential mode passes args and kwargs correctly.
    This ensures argument handling works without threading.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,), (2,)], ["id"])

    # and I have a fdtf function that uses args and kwargs
    @fdtf(output_schema="result INT", max_workers=None)
    def compute(self, row, multiplier, offset=0):  # noqa: ARG001
        return (row["id"] * multiplier + offset,)

    # and I call it with args and kwargs
    result_df = compute(df, 5, offset=10)
    results = sorted(result_df.collect(), key=lambda r: r["id"])

    # then args and kwargs should be passed correctly
    assert results[0]["result"] == 15  # 1 * 5 + 10
    assert results[1]["result"] == 20  # 2 * 5 + 10


# Signature Validation Tests


@pytest.mark.spark40_only
def test_validate_fdtf_signature_with_row_only():
    """
    Test that _validate_fdtf_signature accepts functions with only row parameter.
    This ensures the simple signature works without init_fn.
    """

    # when I have a function with only row parameter
    def simple_fn(row):
        return row

    # and I validate it without init_fn
    result = _validate_fdtf_signature(simple_fn, has_init_fn=False)

    # then it should return False (no context needed)
    assert result is False


@pytest.mark.spark40_only
def test_validate_fdtf_signature_with_self_and_row():
    """
    Test that _validate_fdtf_signature accepts functions with self and row parameters.
    This ensures the context signature is recognized.
    """

    # when I have a function with self and row parameters
    def context_fn(self, row):
        return row

    # and I validate it without init_fn
    result = _validate_fdtf_signature(context_fn, has_init_fn=False)

    # then it should return True (uses context because first param is 'self')
    assert result is True


@pytest.mark.spark40_only
def test_validate_fdtf_signature_with_init_fn_requires_self():
    """
    Test that _validate_fdtf_signature requires self when init_fn is provided.
    This ensures proper validation when resources are initialized.
    """

    # when I have a function with only row parameter
    def simple_fn(row):
        return row

    # and I validate it with init_fn=True
    # then it should raise TypeError
    with pytest.raises(TypeError) as exc_info:
        _validate_fdtf_signature(simple_fn, has_init_fn=True)

    assert "must accept 'self' as the first parameter" in str(exc_info.value)
    assert "simple_fn" in str(exc_info.value)


@pytest.mark.spark40_only
def test_validate_fdtf_signature_with_no_params_raises_error():
    """
    Test that _validate_fdtf_signature raises error for functions with no parameters.
    This ensures proper error messages for invalid signatures.
    """

    # when I have a function with no parameters
    def no_params():
        return 42

    # and I validate it
    # then it should raise TypeError
    with pytest.raises(TypeError) as exc_info:
        _validate_fdtf_signature(no_params, has_init_fn=False)

    assert "must accept at least a 'row' parameter" in str(exc_info.value)
    assert "no_params" in str(exc_info.value)


@pytest.mark.spark40_only
def test_validate_fdtf_signature_with_init_fn_and_self_succeeds():
    """
    Test that _validate_fdtf_signature succeeds when init_fn is provided and function has self.
    This ensures the expected use case works.
    """

    # when I have a function with self and row parameters
    def context_fn(self, row):
        return row

    # and I validate it with init_fn=True
    result = _validate_fdtf_signature(context_fn, has_init_fn=True)

    # then it should return True (uses context)
    assert result is True


@pytest.mark.spark40_only
def test_fdtf_without_self_parameter(spark):
    """
    Test that fdtf works with simple function signature (no self).
    This ensures the simplified API works end-to-end.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,), (2,), (3,)], ["id"])

    # and I have a fdtf function without self parameter
    @fdtf(output_schema="doubled INT", max_workers=2)
    def double_id(row):
        return (row["id"] * 2,)

    # and I apply the function
    result_df = double_id(df)
    results = sorted(result_df.collect(), key=lambda r: r["id"])

    # then it should work correctly
    assert len(results) == 3
    assert results[0]["doubled"] == 2
    assert results[1]["doubled"] == 4
    assert results[2]["doubled"] == 6


@pytest.mark.spark40_only
def test_fdtf_without_self_with_args(spark):
    """
    Test that fdtf without self correctly passes positional arguments.
    This ensures argument handling works with the simplified signature.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,), (2,)], ["id"])

    # and I have a fdtf function without self that uses args
    @fdtf(output_schema="result INT", max_workers=2)
    def add_values(row, arg1, arg2):
        return (row["id"] + arg1 + arg2,)

    # and I call it with positional arguments
    result_df = add_values(df, 10, 20)
    results = sorted(result_df.collect(), key=lambda r: r["id"])

    # then arguments should be passed correctly
    assert results[0]["result"] == 31  # 1 + 10 + 20
    assert results[1]["result"] == 32  # 2 + 10 + 20


@pytest.mark.spark40_only
def test_fdtf_without_self_with_kwargs(spark):
    """
    Test that fdtf without self correctly passes keyword arguments.
    This ensures kwarg handling works with the simplified signature.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,), (2,)], ["id"])

    # and I have a fdtf function without self that uses kwargs
    @fdtf(output_schema="result INT", max_workers=2)
    def compute(row, multiplier=1, offset=0):
        return (row["id"] * multiplier + offset,)

    # and I call it with keyword arguments
    result_df = compute(df, multiplier=5, offset=10)
    results = sorted(result_df.collect(), key=lambda r: r["id"])

    # then kwargs should be passed correctly
    assert results[0]["result"] == 15  # 1 * 5 + 10
    assert results[1]["result"] == 20  # 2 * 5 + 10


@pytest.mark.spark40_only
def test_fdtf_with_init_fn_requires_self_in_decorated_function(spark):
    """
    Test that fdtf raises error when init_fn is provided but function lacks self.
    This ensures helpful error messages for misconfigured functions.
    """

    # when I have an init function
    def my_init(self):
        self.value = 42

    # and I try to decorate a function without self
    # then it should raise TypeError at decoration time
    with pytest.raises(TypeError) as exc_info:

        @fdtf(output_schema="result INT", init_fn=my_init)
        def missing_self(row):
            return (row["id"],)

    assert "must accept 'self' as the first parameter" in str(exc_info.value)


@pytest.mark.spark40_only
def test_fdtf_without_self_sequential_execution(spark):
    """
    Test that fdtf without self works with sequential execution.
    This ensures the simplified signature works with max_workers=None.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,), (2,), (3,)], ["id"])

    # and I have a fdtf function without self and max_workers=None
    @fdtf(output_schema="squared INT", max_workers=None)
    def square_id(row):
        return (row["id"] ** 2,)

    # and I apply the function
    result_df = square_id(df)
    results = result_df.collect()

    # then it should work correctly and preserve order
    assert results[0]["id"] == 1
    assert results[0]["squared"] == 1
    assert results[1]["id"] == 2
    assert results[1]["squared"] == 4
    assert results[2]["id"] == 3
    assert results[2]["squared"] == 9


@pytest.mark.spark40_only
def test_fdtf_without_self_error_handling(spark):
    """
    Test that fdtf without self correctly captures errors.
    This ensures error handling works with the simplified signature.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,), (2,)], ["id"])

    # and I have a fdtf function without self that raises an error
    @fdtf(output_schema="result INT", max_workers=2)
    def may_fail(row):
        if row["id"] == 1:
            raise ValueError("Simulated error")
        return (row["id"] * 10,)

    # and I apply the function
    result_df = may_fail(df)
    results = sorted(result_df.collect(), key=lambda r: r["id"])

    # then the error row should have null result and error in metadata
    assert results[0]["result"] is None
    assert_failure(results[0])
    error = get_last_error(results[0])
    assert "ValueError" in error
    assert "Simulated error" in error

    # and the success row should have result and no error
    assert results[1]["result"] == 20
    assert_success(results[1])


# Metadata Column Disabled Tests


@pytest.mark.spark40_only
def test_fdtf_with_metadata_column_none_disables_metadata(spark):
    """
    Test that fdtf with metadata_column=None produces output without metadata.
    This ensures metadata can be disabled for simpler output.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,), (2,), (3,)], ["id"])

    # and I have a fdtf function with metadata_column=None
    @fdtf(output_schema="doubled INT", max_workers=None, metadata_column=None)
    def double_id(row):
        return (row["id"] * 2,)

    # and I apply the function
    result_df = double_id(df)
    results = result_df.collect()

    # then I should get results without metadata column
    assert len(results) == 3
    expected_fields = ["id", "doubled"]
    actual_fields = [f.name for f in result_df.schema.fields]
    assert actual_fields == expected_fields

    # and results should be correct
    assert results[0]["doubled"] == 2
    assert results[1]["doubled"] == 4
    assert results[2]["doubled"] == 6


@pytest.mark.spark40_only
def test_fdtf_with_metadata_column_none_with_concurrent_execution(spark):
    """
    Test that fdtf with metadata_column=None works with concurrent execution.
    This ensures the feature works in all execution modes.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,), (2,), (3,)], ["id"])

    # and I have a fdtf function with max_workers and metadata_column=None
    @fdtf(output_schema="doubled INT", max_workers=2, metadata_column=None)
    def double_id(row):
        return (row["id"] * 2,)

    # and I apply the function
    result_df = double_id(df)
    results = sorted(result_df.collect(), key=lambda r: r["id"])

    # then schema should not have metadata
    expected_fields = ["id", "doubled"]
    actual_fields = [f.name for f in result_df.schema.fields]
    assert actual_fields == expected_fields

    # and results should be correct
    assert results[0]["doubled"] == 2
    assert results[1]["doubled"] == 4
    assert results[2]["doubled"] == 6


@pytest.mark.spark40_only
def test_fdtf_with_metadata_column_none_with_init_fn(spark):
    """
    Test that fdtf with metadata_column=None works with init_fn.
    This ensures the feature is compatible with resource management.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,), (2,)], ["id"])

    # and I have init_fn and metadata_column=None
    def my_init(self):
        self.multiplier = 10

    @fdtf(output_schema="result INT", init_fn=my_init, max_workers=None, metadata_column=None)
    def multiply(self, row):
        return (row["id"] * self.multiplier,)

    # and I apply the function
    result_df = multiply(df)
    results = result_df.collect()

    # then schema should not have metadata
    expected_fields = ["id", "result"]
    actual_fields = [f.name for f in result_df.schema.fields]
    assert actual_fields == expected_fields

    # and results should be correct
    assert results[0]["result"] == 10
    assert results[1]["result"] == 20


@pytest.mark.spark40_only
def test_fdtf_with_metadata_column_none_error_returns_null(spark):
    """
    Test that fdtf with metadata_column=None still returns null on error.
    This ensures error handling works without metadata.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,), (2,)], ["id"])

    # and I have a fdtf function that fails for some rows
    @fdtf(output_schema="result INT", max_workers=None, metadata_column=None)
    def may_fail(row):
        if row["id"] == 1:
            raise ValueError("Simulated error")
        return (row["id"] * 10,)

    # and I apply the function
    result_df = may_fail(df)
    results = result_df.collect()

    # then schema should not have metadata
    expected_fields = ["id", "result"]
    actual_fields = [f.name for f in result_df.schema.fields]
    assert actual_fields == expected_fields

    # and the error row should have null result
    assert results[0]["result"] is None

    # and the success row should have result
    assert results[1]["result"] == 20


# Generator/Yield Tests (Exploding Rows)


@pytest.mark.spark40_only
def test_fdtf_with_generator_yields_multiple_rows(spark):
    """
    Test that fdtf supports generators that yield multiple rows per input.
    This ensures exploding behavior works correctly.
    """
    # when I have a DataFrame with single row
    df = spark.createDataFrame([(1, "a")], ["id", "value"])

    # and I have a fdtf function that yields multiple rows
    @fdtf(output_schema="iteration INT", max_workers=None, metadata_column=None)
    def explode_rows(row):  # noqa: ARG001
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


@pytest.mark.spark40_only
def test_fdtf_with_generator_multiple_input_rows(spark):
    """
    Test that fdtf with generator works correctly with multiple input rows.
    This ensures each input row is expanded independently.
    """
    # when I have a DataFrame with multiple rows
    df = spark.createDataFrame([(1,), (2,)], ["id"])

    # and I have a fdtf function that yields multiple rows based on id
    @fdtf(output_schema="value INT", max_workers=None, metadata_column=None)
    def explode_by_id(row):
        for i in range(row["id"]):
            yield (i,)

    # and I apply the function
    result_df = explode_by_id(df)
    results = sorted(result_df.collect(), key=lambda r: (r["id"], r["value"]))

    # then row with id=1 should have 1 output, id=2 should have 2 outputs
    assert len(results) == 3  # 1 + 2
    assert results[0]["id"] == 1
    assert results[0]["value"] == 0
    assert results[1]["id"] == 2
    assert results[1]["value"] == 0
    assert results[2]["id"] == 2
    assert results[2]["value"] == 1


@pytest.mark.spark40_only
def test_fdtf_with_generator_and_metadata(spark):
    """
    Test that fdtf with generator includes metadata on each yielded row.
    This ensures metadata is shared across all rows from same input.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,)], ["id"])

    # and I have a fdtf function that yields multiple rows with metadata enabled
    @fdtf(output_schema="value INT", max_workers=None)
    def explode_rows(row):  # noqa: ARG001
        for i in range(2):
            yield (i,)

    # and I apply the function
    result_df = explode_rows(df)
    results = result_df.collect()

    # then each output row should have metadata
    assert len(results) == 2
    assert "_metadata" in [f.name for f in result_df.schema.fields]
    # Both rows should have the same metadata (same attempt info)
    assert results[0]["_metadata"] == results[1]["_metadata"]
    assert len(results[0]["_metadata"]) == 1  # One successful attempt


@pytest.mark.spark40_only
def test_fdtf_with_generator_concurrent_execution(spark):
    """
    Test that fdtf with generator works with concurrent execution.
    This ensures thread pool handles multiple results per input.
    """
    # when I have a DataFrame with multiple rows
    df = spark.createDataFrame([(1,), (2,), (3,)], ["id"])

    # and I have a fdtf function that yields multiple rows
    @fdtf(output_schema="doubled INT", max_workers=2, metadata_column=None)
    def explode_doubled(row):
        yield (row["id"] * 2,)
        yield (row["id"] * 2 + 1,)

    # and I apply the function
    result_df = explode_doubled(df)
    results = result_df.collect()

    # then I should get 6 rows (2 per input)
    assert len(results) == 6

    # and results should contain expected values
    doubled_values = sorted([r["doubled"] for r in results])
    assert doubled_values == [2, 3, 4, 5, 6, 7]


@pytest.mark.spark40_only
def test_fdtf_with_generator_empty_yield(spark):
    """
    Test that fdtf handles generators that yield nothing for some rows.
    This ensures filtering behavior works.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,), (2,), (3,)], ["id"])

    # and I have a fdtf function that only yields for even ids
    @fdtf(output_schema="value INT", max_workers=None, metadata_column=None)
    def filter_even(row):
        if row["id"] % 2 == 0:
            yield (row["id"],)
        # odd ids yield nothing

    # and I apply the function
    result_df = filter_even(df)
    results = result_df.collect()

    # then only even rows should be in output
    assert len(results) == 1
    assert results[0]["id"] == 2
    assert results[0]["value"] == 2


@pytest.mark.spark40_only
def test_fdtf_with_generator_error_retries(spark):
    """
    Test that fdtf retries the entire generator on error.
    This ensures retry logic works with generators.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,)], ["id"])

    # and I have a function that tracks attempts and fails initially
    def my_init(self):
        self.attempt_count = 0

    @fdtf(output_schema="value INT", init_fn=my_init, max_workers=None, max_retries=2)
    def flaky_generator(self, row):  # noqa: ARG001
        self.attempt_count += 1
        if self.attempt_count < 2:
            raise Exception("temporary failure")
        yield (1,)
        yield (2,)

    # and I apply the function
    result_df = flaky_generator(df)
    results = result_df.collect()

    # then it should eventually succeed with all yielded values
    assert len(results) == 2
    assert results[0]["value"] == 1
    assert results[1]["value"] == 2

    # and metadata should show retry attempts
    metadata = results[0]["_metadata"]
    assert len(metadata) == 2  # 1 failure + 1 success


@pytest.mark.spark40_only
def test_fdtf_mixed_return_and_yield_single_value(spark):
    """
    Test that fdtf handles single return values alongside generator capability.
    This ensures backwards compatibility with return statements.
    """
    # when I have a DataFrame
    df = spark.createDataFrame([(1,), (2,)], ["id"])

    # and I have a fdtf function that returns (not yields)
    @fdtf(output_schema="doubled INT", max_workers=None, metadata_column=None)
    def double_value(row):
        return (row["id"] * 2,)

    # and I apply the function
    result_df = double_value(df)
    results = sorted(result_df.collect(), key=lambda r: r["id"])

    # then it should work as before (one output per input)
    assert len(results) == 2
    assert results[0]["doubled"] == 2
    assert results[1]["doubled"] == 4
