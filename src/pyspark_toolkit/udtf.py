import inspect
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from types import GeneratorType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Concatenate,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    ParamSpec,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
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

# Default column name for fdtf metadata
DEFAULT_METADATA_COLUMN = "_metadata"


def _validate_fdtf_signature(fn: Callable, has_init_fn: bool) -> bool:
    """Validate the function signature and determine if it uses context.

    Args:
        fn: The decorated function
        has_init_fn: Whether init_fn was provided to fdtf

    Returns:
        True if function expects context (self) as first param, False otherwise

    Raises:
        TypeError: If the signature is invalid for the given configuration
    """
    sig = inspect.signature(fn)
    params = [
        p
        for p in sig.parameters.values()
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    num_params = len(params)

    if num_params == 0:
        raise TypeError(
            f"Function '{fn.__name__}' must accept at least a 'row' parameter. "
            f"Expected signature: def {fn.__name__}(row, ...) or def {fn.__name__}(self, row, ...)"
        )

    if has_init_fn:
        if num_params < 2:
            raise TypeError(
                f"When using init_fn, function '{fn.__name__}' must accept 'self' as the first parameter "
                f"to access initialized resources. "
                f"Expected signature: def {fn.__name__}(self, row, ...)"
            )
        return True  # Uses context

    # No init_fn - check first param name to determine intent
    first_param_name = params[0].name
    if first_param_name == "self":
        # User explicitly wants context even without init_fn
        return True
    elif num_params >= 1:
        # Assume first param is row
        return False

    return False


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


@runtime_checkable
class FdtfContext(Protocol):
    """
    Protocol for the context object passed to fdtf functions.

    Your init_fn should set attributes on this object, which will then be
    accessible in your processing function via the first parameter (conventionally
    named ``self``). All attributes are dynamically typed as Any.

    This is a Protocol (structural type), not a concrete class. The actual
    context object is created internally by fdtf. Any object with dynamic
    attribute access satisfies this protocol.

    Example:
        def my_init(self: FdtfContext) -> None:
            self.http = httpx.Client(timeout=30)
            self.api_key = "secret"

        @fdtf(...)
        def call_api(self: FdtfContext, row: dict) -> tuple:
            return (self.http.get(row["url"]).text,)
    """

    def __getattr__(self, name: str) -> Any: ...
    def __setattr__(self, name: str, value: Any) -> None: ...


def fdtf(
    *,
    returnType: Union[StructType, str],
    init_fn: Optional[Callable[["FdtfContext"], None]] = None,
    cleanup_fn: Optional[Callable[["FdtfContext"], None]] = None,
    max_workers: Optional[int] = 20,
    max_retries: int = 0,
    metadata_column: Optional[str] = DEFAULT_METADATA_COLUMN,
    with_single_partition: bool = False,
) -> Callable[[Callable[Concatenate["FdtfContext", RowDict, P], R]], DataFrameTransformer[P]]:
    """
    Flexible DataFrame Table Function - processes rows with optional concurrency using a thread pool.

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
        returnType: Schema for output columns (StructType or DDL string like "col1 INT, col2 STRING").
                    Named to match PySpark's @udtf decorator parameter.
        init_fn: Optional. Called once per partition to initialize resources. Receives a FdtfContext
                 object - set attributes on it (e.g., self.http = httpx.Client()).
        cleanup_fn: Optional. Called in terminate() to close resources. Receives the same FdtfContext.
        max_workers: Thread pool size per partition (default: 20). Set to None or 0 for sequential
                     execution without threading.
        max_retries: Number of retry attempts on failure (default: 0, no retries)
        metadata_column: Name for the metadata column (default: "_metadata"). Set to None to
                         disable metadata tracking entirely.
        with_single_partition: Force all data to single partition (default: False)

    Function signatures:
        - With init_fn: def fn(self: FdtfContext, row: dict, *args, **kwargs) -> tuple
        - Without init_fn: def fn(row: dict, *args, **kwargs) -> tuple

    When init_fn is provided, your function MUST accept 'self' as the first parameter
    to access initialized resources. Without init_fn, 'self' is optional.

    Return values:
        - Single value: return value  (wrapped in tuple automatically)
        - Tuple: return (val1, val2)  (used as-is)
        - Generator: yield (val1,); yield (val2,)  (multiple output rows per input)

    Example with init/cleanup and concurrency:
        def my_init(self: FdtfContext) -> None:
            self.http = httpx.Client(timeout=30)
            self.api_client = SomeAPIClient()

        def my_cleanup(self: FdtfContext) -> None:
            self.http.close()

        @fdtf(
            returnType="result STRING, score FLOAT",
            init_fn=my_init,
            cleanup_fn=my_cleanup,
            max_workers=10,
            max_retries=3,
        )
        def call_api(self, row, api_key):
            response = self.api_client.call(row["input"], key=api_key)
            return (response.text, response.score)

        result_df = call_api(input_df, api_key="secret")

    Example without init_fn (simple signature):
        @fdtf(returnType="doubled INT", max_workers=None)
        def double_value(row):
            return (row["value"] * 2,)

        result_df = double_value(input_df)
    """
    parsed_schema = _parse_schema(returnType)
    has_init_fn = init_fn is not None
    include_metadata = metadata_column is not None

    # Build the full output schema: user schema + metadata (array of attempts) if enabled
    fdtf_meta_fields: List[StructField] = []
    if include_metadata:
        fdtf_meta_fields = [_build_metadata_schema(metadata_column)]

    def _decorate(fn: Callable[..., R]) -> DataFrameTransformer[P]:
        # Validate signature and determine if function uses context
        uses_context = _validate_fdtf_signature(fn, has_init_fn)

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
            captured_uses_context = uses_context
            captured_include_metadata = include_metadata
            num_user_output_cols = len(user_output_cols)

            # Define helper functions locally to avoid cloudpickle module reference issues.
            # When cloudpickle serializes the UDTF class, it would otherwise record these
            # as module references (e.g., pyspark_toolkit.udtf._as_dict). On unpickling,
            # Python would try to import pyspark_toolkit, which fails in Databricks Connect
            # because withDependencies() installs packages at query execution time, but UDTF
            # analysis happens before execution begins.

            class _LocalFdtfContext:
                """Local context class to avoid cloudpickle module reference issues."""

                if TYPE_CHECKING:

                    def __getattr__(self, name: str) -> Any: ...
                    def __setattr__(self, name: str, value: Any) -> None: ...

            def _local_as_dict(r):
                return r.asDict(recursive=True) if hasattr(r, "asDict") else dict(r)

            def _local_ensure_tuple(value: Any) -> Tuple[Any, ...]:
                if isinstance(value, tuple):
                    return value
                return (value,)

            def _local_is_generator(value: Any) -> bool:
                return isinstance(value, (GeneratorType, Iterator)) and not isinstance(value, (str, bytes))

            def _local_normalize_result(value: Any) -> Iterable[Tuple[Any, ...]]:
                if _local_is_generator(value):
                    for item in value:
                        yield _local_ensure_tuple(item)
                else:
                    yield _local_ensure_tuple(value)

            @dataclass
            class _Analyze(AnalyzeResult):
                pass

            @udtf
            class _FdtfUDTF:
                def __init__(self, analyze_result: Optional[AnalyzeResult] = None):
                    self.ctx = _LocalFdtfContext()
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
                    combined = StructType(list(base_schema.fields) + list(parsed_schema.fields) + fdtf_meta_fields)
                    return _Analyze(schema=combined, withSinglePartition=with_single_partition)

                def _execute_with_retry(
                    self,
                    row_dict: Dict[str, Any],
                    arg_vals: List[Any],
                    kw_vals: Dict[str, Any],
                ) -> Tuple[Optional[List[Tuple[Any, ...]]], List[AttemptRecord]]:
                    """Execute the function with optional retry logic.

                    Returns: (list of result tuples or None, list of attempt records)
                    Each attempt record is (attempt, started_at, duration_ms, error)

                    Supports both single returns and generators - generators yield multiple
                    result tuples, single returns yield one.
                    """
                    max_attempts = captured_max_retries + 1
                    attempts: List[AttemptRecord] = []

                    for attempt in range(1, max_attempts + 1):
                        started_at = datetime.now(timezone.utc)
                        start_time = time.perf_counter()

                        try:
                            if captured_uses_context:
                                result = captured_fn(self.ctx, row_dict, *arg_vals, **kw_vals)  # type: ignore[call-arg]
                            else:
                                result = captured_fn(row_dict, *arg_vals, **kw_vals)  # type: ignore[call-arg]

                            # Normalize result to list of tuples (handles generators and single values)
                            results = list(_local_normalize_result(result))

                            duration_ms = int((time.perf_counter() - start_time) * 1000)
                            # Success: record attempt with no error
                            attempts.append((attempt, started_at, duration_ms, None))
                            return results, attempts

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
                ) -> List[Tuple[Any, ...]]:
                    """Process a single row and return list of full output tuples.

                    Returns a list because generators can yield multiple rows per input.
                    """
                    results, attempts = self._execute_with_retry(row_dict, arg_vals, kw_vals)

                    if results is None:
                        # Error case: null for all user output columns, single row
                        null_results = tuple(None for _ in range(num_user_output_cols))
                        if captured_include_metadata:
                            return [base_vals + null_results + (attempts,)]
                        else:
                            return [base_vals + null_results]
                    else:
                        # Success case: may have multiple results from generator
                        output_rows = []
                        for result in results:
                            if captured_include_metadata:
                                output_rows.append(base_vals + result + (attempts,))
                            else:
                                output_rows.append(base_vals + result)
                        return output_rows

                def eval(self, row):
                    d = _local_as_dict(row)
                    arg_vals = [d.get(a) for a in arg_names]
                    kw_vals = {k: d.get(k) for k in kwarg_names}
                    base_vals = tuple(d.get(c) for c in base_cols)

                    if self.pool is None or self.futs is None:
                        # Sequential execution - process and yield immediately
                        for output_row in self._process_one(base_vals, d, arg_vals, kw_vals):
                            yield output_row
                    else:
                        # Concurrent execution - submit to thread pool
                        fut = self.pool.submit(self._process_one, base_vals, d, arg_vals, kw_vals)
                        self.futs.append(fut)

                        # Yield results as they complete to avoid memory buildup
                        if captured_max_workers is not None and len(self.futs) >= captured_max_workers:
                            for f in as_completed(self.futs[:captured_max_workers]):
                                for output_row in f.result():
                                    yield output_row
                            self.futs = self.futs[captured_max_workers:]

                def terminate(self):
                    if self.pool is not None and self.futs is not None:
                        # Drain remaining futures
                        for f in as_completed(self.futs):
                            for output_row in f.result():
                                yield output_row

                        # Shutdown thread pool
                        self.pool.shutdown(wait=True, cancel_futures=False)

                    # Cleanup resources
                    if captured_cleanup_fn is not None:
                        captured_cleanup_fn(self.ctx)

            fn_name = f"udtf_{str(uuid.uuid4()).replace('-', '')}"
            input_df.sparkSession.udtf.register(fn_name, _FdtfUDTF)  # type: ignore

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

        _runner.__name__ = getattr(fn, "__name__", "fdtf_runner")
        _runner.__doc__ = fn.__doc__
        return cast(DataFrameTransformer[P], _runner)

    return _decorate
