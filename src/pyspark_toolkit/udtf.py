import json
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from pyspark.sql import DataFrame
from pyspark.sql.functions import udtf
from pyspark.sql.types import DataType, StructType
from pyspark.sql.udtf import AnalyzeArgument, AnalyzeResult

RowDict = Dict[str, Any]
RowFn = Callable[..., Iterable[Tuple[Any, ...]]]


def _as_dict(r):
    return r.asDict(recursive=True) if hasattr(r, "asDict") else dict(r)


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


def fdtf(*, output_schema: Union[StructType, str], with_single_partition: bool = False):
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

    def _decorate(fn: RowFn) -> Callable:
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
                    yield from (base_vals + out for out in fn(d, *arg_vals, **kw_vals))

            fn_name = f"udtf_{str(uuid.uuid4()).replace('-', '')}"
            input_df.sparkSession.udtf.register(fn_name, _AppendColsUDTF)

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
        return _runner

    return _decorate
