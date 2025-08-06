from __future__ import annotations

import pyspark
import pyspark.sql.functions as F
from pyspark.sql import Column

from pyspark_utils.types import ByteColumn, LongColumn, StringColumn


def safe_cast(col: Column, data_type: str) -> Column:
    """
    Version-aware casting that uses try_cast in PySpark 4.0+ and cast in earlier versions.
    """
    pyspark_version = tuple(int(x) for x in pyspark.__version__.split(".")[:2])

    if pyspark_version >= (4, 0):
        return col.try_cast(data_type)  # type: ignore (pyspark 4.0+ has a try_cast)
    else:
        return col.cast(data_type)  # type: ignore (pyspark 3.0- has a cast)


def chars_to_int(col: ByteColumn | StringColumn) -> LongColumn:
    """
    Take a string, encode it as utf-8, and convert those bytes to a bigint

    Currently blows up if our string is too big
    """
    return LongColumn(safe_cast(F.conv(F.hex(col), 16, 10), "bigint"))


def pad_key(key: ByteColumn, block_size: int) -> ByteColumn:
    return ByteColumn(F.rpad(key, block_size, bytes([0])))  # type: ignore (we need to pass bytes to rpad)


def sha2_binary(col: ByteColumn, num_bits: int) -> ByteColumn:
    return ByteColumn(F.to_binary(F.sha2(col, num_bits), F.lit("hex")))
