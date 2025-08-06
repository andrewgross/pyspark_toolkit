from __future__ import annotations

from typing import NewType

import pyspark
import pyspark.sql.functions as F
from pyspark.sql import Column

BooleanColumn = NewType("BooleanColumn", Column)
ByteColumn = NewType("ByteColumn", Column)
IntegerColumn = NewType("IntegerColumn", Column)
LongColumn = NewType("LongColumn", Column)
StringColumn = NewType("StringColumn", Column)
HexStringColumn = NewType("HexStringColumn", Column)


def safe_cast(col: Column, data_type: str) -> Column:
    """
    Version-aware casting that uses try_cast in PySpark 4.0+ and cast in earlier versions.
    """
    pyspark_version = tuple(int(x) for x in pyspark.__version__.split(".")[:2])

    if pyspark_version >= (4, 0):
        return col.try_cast(data_type)
    else:
        return col.cast(data_type)


def chars_to_int(col: StringColumn) -> LongColumn:
    """
    Take a string, encode it as utf-8, and convert those bytes to a bigint

    Currently blows up if our string is too big
    """
    return LongColumn(safe_cast(F.conv(F.hex(col), 16, 10), "bigint"))


def pad_key(key: ByteColumn, block_size: int) -> ByteColumn:
    return ByteColumn(F.rpad(key, block_size, bytes([0])))


def sha2_binary(col: ByteColumn, num_bits: int) -> ByteColumn:
    return ByteColumn(F.to_binary(F.sha2(col, num_bits), F.lit("hex")))
