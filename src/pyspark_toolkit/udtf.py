import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Concatenate,
    Dict,
    Iterable,
    List,
    Optional,
    ParamSpec,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from pyspark.sql import DataFrame
from pyspark.sql.functions import udtf
from pyspark.sql.types import (
    DataType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from pyspark.sql.udtf import AnalyzeArgument, AnalyzeResult

# Type variables for decorator signatures
P = ParamSpec("P")
R = TypeVar("R")

# Type alias for the transformed function signature
DataFrameTransformer = Callable[Concatenate[DataFrame, P], DataFrame]

RowDict = Dict[str, Any]
RowFn = Callable[..., Iterable[Tuple[Any, ...]]]
ConcurrentRowFn = Callable[..., Tuple[Any, ...] | Any]
AttemptRecord = Tuple[int, datetime, int, Optional[str]]

# Default column name for cdtf metadata
DEFAULT_METADATA_COLUMN = "_metadata"


def _build_metadata_schema(column_name: str) -> StructField:
    """Build the schema for the metadata column.

    The metadata column is an array of attempt records, one per attempt.
    Each attempt record contains:
        - attempt: which attempt number (1-indexed)
        - started_at: when the attempt started
        - duration_ms: how long the attempt took
        - error: error message if failed, null if succeeded
    """
    from pyspark.sql.types import ArrayType

    attempt_struct = StructType(
        [
            StructField("attempt", IntegerType(), nullable=False),
            StructField("started_at", TimestampType(), nullable=False),
            StructField("duration_ms", LongType(), nullable=False),
            StructField("error", StringType(), nullable=True),
        ]
    )
    return StructField(column_name, ArrayType(attempt_struct), nullable=False)


def _as_dict(r):
    return r.asDict(recursive=True) if hasattr(r, "asDict") else dict(r)


def _ensure_tuple(value: Any) -> Tuple[Any, ...]:
    """Normalize a return value to a tuple.

    If the value is already a tuple, return it as-is.
    Otherwise, wrap it in a single-element tuple.
    """
    if isinstance(value, tuple):
        return value
    return (value,)


def _parse_schema(schema: Union[StructType, str]) -> StructType:
    """
    Parse a schema from either a StructType or a DDL-formatted string.

    Args:
        schema: Either a StructType object or a DDL string like "col1 INT, col2 STRING"

    Returns:
        A StructType representing the schema

    Raises:
        TypeError: If the schema is not a StructType or string
        ValueError: If the DDL string does not parse to a StructType
    """
    if isinstance(schema, StructType):
        return schema

    if isinstance(schema, str):
        parsed = DataType.fromDDL(schema)
        if not isinstance(parsed, StructType):
            raise ValueError(
                f"Schema DDL must define a struct type, got {type(parsed).__name__}. "
                f"Use format like 'col1 INT, col2 STRING' for multiple columns."
            )
        return parsed

    raise TypeError(f"output_schema must be a StructType or DDL string, got {type(schema).__name__}")


def fdtf(
    *, output_schema: Union[StructType, str], with_single_partition: bool = False
) -> Callable[[Callable[Concatenate[RowDict, P], Iterable[R]]], DataFrameTransformer[P]]:
    """
    Decorator for flexible UDTFs that append new columns to the input DataFrame.

    Your function should expect `row` as the first argument, which is a dict of the values for the current row.

    The return from this function will incorporate all of the existing row values plus any values returned from your function.

    Supports both *args and **kwargs at call time:
        @fdtf(output_schema=StructType([...]))
        def fn(row, *args, **kwargs): ...
        result = fn(df, "foo", 123, named="bar")

    The output_schema parameter accepts either:
        - A StructType object: StructType([StructField("col", IntegerType())])
        - A DDL string: "col INT" or "col1 INT, col2 STRING"
    """
    parsed_schema = _parse_schema(output_schema)

    def _decorate(fn: Callable[Concatenate[RowDict, P], Iterable[R]]) -> DataFrameTransformer[P]:
        def _runner(input_df: DataFrame, *args: Any, **kwargs: Any) -> DataFrame:
            base_schema: StructType = input_df.schema
            base_cols: List[str] = [f.name for f in base_schema.fields]

            # Generate placeholder names for positional args
            arg_names = [f"_arg{i}" for i in range(len(args))]
            kwarg_names = list(kwargs.keys())

            @dataclass
            class _Analyze(AnalyzeResult):
                pass

            @udtf
            class _AppendColsUDTF:
                def __init__(self, analyze_result: Optional[AnalyzeResult] = None):
                    pass

                @staticmethod
                def analyze(table: AnalyzeArgument) -> AnalyzeResult:
                    if not table.isTable:
                        raise Exception("First argument must be TABLE(...)")
                    combined = StructType(list(base_schema.fields) + list(parsed_schema.fields))
                    return _Analyze(schema=combined, withSinglePartition=with_single_partition)

                def eval(self, row):
                    d = _as_dict(row)
                    arg_vals = [d.get(a) for a in arg_names]
                    kw_vals = {k: d.get(k) for k in kwarg_names}
                    base_vals = tuple(d.get(c) for c in base_cols)
                    # always call with both *args and **kwargs
                    for out in fn(d, *arg_vals, **kw_vals):  # type: ignore[call-arg]
                        yield base_vals + _ensure_tuple(out)

            fn_name = f"udtf_{str(uuid.uuid4()).replace('-', '')}"
            input_df.sparkSession.udtf.register(fn_name, _AppendColsUDTF)  # type: ignore

            table_uuid = f"table_{str(uuid.uuid4()).replace('-', '')}"
            input_df.createOrReplaceTempView(table_uuid)

            # Build arg/kwarg projections using named bindings
            # e.g. ":_arg0 AS _arg0, :name AS name"
            all_arg_names = arg_names + kwarg_names
            all_placeholders = [f":{n} AS {n}" for n in all_arg_names]
            proj = ",\n".join(all_placeholders)
            proj = (proj + ",\n") if proj else ""

            sql = f"""
              SELECT *
              FROM {fn_name}(
                TABLE(
                  SELECT
                    {proj}*
                  FROM {table_uuid}
                )
              )
            """
            # bind all args + kwargs into a single dict
            args_dict = {n: v for n, v in zip(arg_names, args)}
            args_dict.update(kwargs)
            return input_df.sparkSession.sql(sql, args=args_dict)

        _runner.__name__ = getattr(fn, "__name__", "fdtf_runner")
        _runner.__doc__ = fn.__doc__
        return cast(DataFrameTransformer[P], _runner)

    return _decorate


class CdtfContext:
    """
    Context object passed to cdtf functions.

    Your init_fn should set attributes on this object, which will then be
    accessible in your processing function via self.

    Example:
        def my_init(self):
            self.http = httpx.Client(timeout=30)
            self.api_client = SomeAPIClient()

        def my_cleanup(self):
            self.http.close()

        @cdtf(...)
        def call_api(self, row, api_key):
            return (self.api_client.call(row["input"], key=api_key),)
    """

    pass


def cdtf(
    *,
    output_schema: Union[StructType, str],
    init_fn: Optional[Callable[["CdtfContext"], None]] = None,
    cleanup_fn: Optional[Callable[["CdtfContext"], None]] = None,
    max_workers: Optional[int] = 20,
    max_retries: int = 0,
    metadata_column: str = DEFAULT_METADATA_COLUMN,
    with_single_partition: bool = False,
) -> Callable[[Callable[Concatenate["CdtfContext", RowDict, P], R]], DataFrameTransformer[P]]:
    """
    DataFrame Table Function - processes rows with optional concurrency using a thread pool.

    This decorator wraps a function that processes individual rows. It can run concurrently
    with a thread pool, or sequentially when max_workers is None or 0. It handles retries
    and provides execution metadata.

    Output schema: input_columns + user_output_columns + metadata_column

    The metadata column is an array of attempt records. Each attempt contains:
        - attempt: Which attempt number (1-indexed)
        - started_at: Timestamp when the attempt started
        - duration_ms: How long the attempt took in milliseconds
        - error: Error message if the attempt failed, null if succeeded

    On success, the array contains records for any failed attempts plus the final successful one.
    On failure (all retries exhausted), the user output columns will be null.

    Args:
        output_schema: Schema for output columns (StructType or DDL string like "col1 INT, col2 STRING")
        init_fn: Optional. Called once per partition to initialize resources. Receives a CdtfContext
                 object - set attributes on it (e.g., self.http = httpx.Client()).
        cleanup_fn: Optional. Called in terminate() to close resources. Receives the same CdtfContext.
        max_workers: Thread pool size per partition (default: 20). Set to None or 0 for sequential
                     execution without threading.
        max_retries: Number of retry attempts on failure (default: 0, no retries)
        metadata_column: Name for the metadata column (default: "_metadata")
        with_single_partition: Force all data to single partition (default: False)

    Your function signature should be:
        def fn(self: CdtfContext, row: dict, *args, **kwargs) -> tuple

    Example with init/cleanup and concurrency:
        def my_init(self):
            self.http = httpx.Client(timeout=30)
            self.api_client = SomeAPIClient()

        def my_cleanup(self):
            self.http.close()

        @cdtf(
            output_schema="result STRING, score FLOAT",
            init_fn=my_init,
            cleanup_fn=my_cleanup,
            max_workers=10,
            max_retries=3,
        )
        def call_api(self, row, api_key):
            response = self.api_client.call(row["input"], key=api_key)
            return (response.text, response.score)

        result_df = call_api(input_df, api_key="secret")

    Example without concurrency (sequential execution):
        @cdtf(output_schema="doubled INT", max_workers=None)
        def double_value(self, row):
            return (row["value"] * 2,)

        result_df = double_value(input_df)
    """
    parsed_schema = _parse_schema(output_schema)

    # Build the full output schema: user schema + metadata (array of attempts)
    cdtf_meta_fields = [
        _build_metadata_schema(metadata_column),
    ]

    def _decorate(fn: Callable[Concatenate["CdtfContext", RowDict, P], R]) -> DataFrameTransformer[P]:
        def _runner(input_df: DataFrame, *args: Any, **kwargs: Any) -> DataFrame:
            base_schema: StructType = input_df.schema
            base_cols: List[str] = [f.name for f in base_schema.fields]
            user_output_cols: List[str] = [f.name for f in parsed_schema.fields]

            # Generate placeholder names for positional args
            arg_names = [f"_arg{i}" for i in range(len(args))]
            kwarg_names = list(kwargs.keys())

            # Capture these for use inside the UDTF class
            captured_init_fn = init_fn
            captured_cleanup_fn = cleanup_fn
            captured_max_retries = max_retries
            captured_max_workers = max_workers
            captured_fn = fn
            num_user_output_cols = len(user_output_cols)

            @dataclass
            class _Analyze(AnalyzeResult):
                pass

            @udtf
            class _ConcurrentUDTF:
                def __init__(self, analyze_result: Optional[AnalyzeResult] = None):
                    self.ctx = CdtfContext()
                    if captured_init_fn is not None:
                        captured_init_fn(self.ctx)

                    # Only create thread pool if concurrent execution is enabled
                    if captured_max_workers:
                        self.pool: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=captured_max_workers)
                        self.futs: Optional[List] = []
                    else:
                        self.pool = None
                        self.futs = None

                @staticmethod
                def analyze(table: AnalyzeArgument) -> AnalyzeResult:
                    if not table.isTable:
                        raise Exception("First argument must be TABLE(...)")
                    combined = StructType(list(base_schema.fields) + list(parsed_schema.fields) + cdtf_meta_fields)
                    return _Analyze(schema=combined, withSinglePartition=with_single_partition)

                def _execute_with_retry(
                    self,
                    row_dict: Dict[str, Any],
                    arg_vals: List[Any],
                    kw_vals: Dict[str, Any],
                ) -> Tuple[Optional[Tuple[Any, ...]], List[AttemptRecord]]:
                    """Execute the function with optional retry logic.

                    Returns: (result_tuple or None, list of attempt records)
                    Each attempt record is (attempt, started_at, duration_ms, error)
                    """
                    max_attempts = captured_max_retries + 1
                    attempts: List[AttemptRecord] = []

                    for attempt in range(1, max_attempts + 1):
                        started_at = datetime.now(timezone.utc)
                        start_time = time.perf_counter()

                        try:
                            result = captured_fn(self.ctx, row_dict, *arg_vals, **kw_vals)  # type: ignore[call-arg]
                            result = _ensure_tuple(result)
                            duration_ms = int((time.perf_counter() - start_time) * 1000)
                            # Success: record attempt with no error
                            attempts.append((attempt, started_at, duration_ms, None))
                            return result, attempts

                        except Exception as e:
                            duration_ms = int((time.perf_counter() - start_time) * 1000)
                            error_msg = f"{type(e).__name__}: {e}"
                            attempts.append((attempt, started_at, duration_ms, error_msg))

                            # Retry if we have attempts remaining
                            if attempt < max_attempts:
                                continue

                            # No more retries - return failure
                            break

                    return None, attempts

                def _process_one(
                    self,
                    base_vals: Tuple[Any, ...],
                    row_dict: Dict[str, Any],
                    arg_vals: List[Any],
                    kw_vals: Dict[str, Any],
                ) -> Tuple[Any, ...]:
                    """Process a single row and return the full output tuple."""
                    result, attempts = self._execute_with_retry(row_dict, arg_vals, kw_vals)

                    if result is None:
                        # Error case: null for all user output columns
                        null_results = tuple(None for _ in range(num_user_output_cols))
                        return base_vals + null_results + (attempts,)
                    else:
                        return base_vals + result + (attempts,)

                def eval(self, row):
                    d = _as_dict(row)
                    arg_vals = [d.get(a) for a in arg_names]
                    kw_vals = {k: d.get(k) for k in kwarg_names}
                    base_vals = tuple(d.get(c) for c in base_cols)

                    if self.pool is None or self.futs is None:
                        # Sequential execution - process and yield immediately
                        yield self._process_one(base_vals, d, arg_vals, kw_vals)
                    else:
                        # Concurrent execution - submit to thread pool
                        fut = self.pool.submit(self._process_one, base_vals, d, arg_vals, kw_vals)
                        self.futs.append(fut)

                        # Yield results as they complete to avoid memory buildup
                        if captured_max_workers is not None and len(self.futs) >= captured_max_workers:
                            for f in as_completed(self.futs[:captured_max_workers]):
                                yield f.result()
                            self.futs = self.futs[captured_max_workers:]

                def terminate(self):
                    if self.pool is not None and self.futs is not None:
                        # Drain remaining futures
                        for f in as_completed(self.futs):
                            yield f.result()

                        # Shutdown thread pool
                        self.pool.shutdown(wait=True, cancel_futures=False)

                    # Cleanup resources
                    if captured_cleanup_fn is not None:
                        captured_cleanup_fn(self.ctx)

            fn_name = f"udtf_{str(uuid.uuid4()).replace('-', '')}"
            input_df.sparkSession.udtf.register(fn_name, _ConcurrentUDTF)  # type: ignore

            table_uuid = f"table_{str(uuid.uuid4()).replace('-', '')}"
            input_df.createOrReplaceTempView(table_uuid)

            # Build arg/kwarg projections using named bindings
            all_arg_names = arg_names + kwarg_names
            all_placeholders = [f":{n} AS {n}" for n in all_arg_names]
            proj = ",\n".join(all_placeholders)
            proj = (proj + ",\n") if proj else ""

            sql = f"""
              SELECT *
              FROM {fn_name}(
                TABLE(
                  SELECT
                    {proj}*
                  FROM {table_uuid}
                )
              )
            """
            args_dict = {n: v for n, v in zip(arg_names, args)}
            args_dict.update(kwargs)
            return input_df.sparkSession.sql(sql, args=args_dict)

        _runner.__name__ = getattr(fn, "__name__", "cdtf_runner")
        _runner.__doc__ = fn.__doc__
        return cast(DataFrameTransformer[P], _runner)

    return _decorate
