from __future__ import annotations

import pyspark
import pyspark.sql.functions as F
from pyspark.sql import Column

from pyspark_toolkit.types import ByteColumn, HexStringColumn, LongColumn, StringColumn


def safe_cast(col: Column, data_type: str) -> Column:
    """
    Version-aware casting that uses try_cast in PySpark 4.0+ and cast in earlier versions.
    """
    pyspark_version = tuple(int(x) for x in pyspark.__version__.split(".")[:2])

    if pyspark_version >= (4, 0):
        return col.try_cast(data_type)  # type: ignore (pyspark 4.0+ has a try_cast)
    else:
        return col.cast(data_type)  # type: ignore (pyspark 3.0- has a cast)


def chars_to_int(col: ByteColumn | StringColumn | HexStringColumn) -> LongColumn:
    """
    Take a string, encode it as utf-8, and convert those bytes to a bigint

    Currently blows up if our string is too big
    """
    return LongColumn(safe_cast(F.conv(F.hex(col), 16, 10), "bigint"))


def pad_key(key: ByteColumn, block_size: int) -> ByteColumn:
    """
    Pads the key with 0s to the block size
    Args:
        key: The key to pad
        block_size: The block size to pad to
    Returns:
        The padded key as a ByteColumn
    """
    return ByteColumn(F.rpad(key, block_size, bytes([0])))  # type: ignore (we need to pass bytes to rpad)


def sha2_binary(col: ByteColumn, num_bits: int) -> ByteColumn:
    """
    Converts the column to a binary representation of the SHA-2 hash
    Args:
        col: The column to convert
        num_bits: The number of bits to use for the SHA-2 hash
    Returns:
        The binary representation of the SHA-2 hash
    """
    return ByteColumn(F.to_binary(F.sha2(col, num_bits), F.lit("hex")))


def split_last_chars(col: StringColumn) -> HexStringColumn:
    """
    Splits the last 4 characters of the column.
    Args:
        col: The column to split
    Returns:
        The last 4 characters of the column
    """
    return HexStringColumn(F.substring(col, -4, 4))


def map_concat(*cols: Column) -> Column:
    """
    Concatenates multiple map columns with right-override merge strategy.

    When duplicate keys exist across maps, values from rightmost maps take precedence.
    This mimics PySpark's map_concat function but with built-in LAST_WIN behavior
    without requiring spark configuration changes.

    Args:
        *cols: Variable number of map columns to concatenate

    Returns:
        Column containing the concatenated map with rightmost values for duplicate keys
    """
    if not cols:
        return F.map_from_arrays(F.array(), F.array())

    if len(cols) == 1:
        return cols[0]

    # Start with the first map
    result = cols[0]

    # Merge each subsequent map, with later maps overriding earlier ones
    for next_map in cols[1:]:
        # Get all keys from both maps
        result_keys = F.map_keys(result)
        next_keys = F.map_keys(next_map)
        all_keys = F.array_distinct(F.array_union(result_keys, next_keys))

        # For each key, take value from next_map if it exists, otherwise from result
        values = F.transform(all_keys, lambda key: F.coalesce(F.element_at(next_map, key), F.element_at(result, key)))

        result = F.map_from_arrays(all_keys, values)

    return result
