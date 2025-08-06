from typing import Annotated

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from pyspark_toolkit.types import (
    BooleanColumn,
    HexStringColumn,
    IntegerColumn,
    StringColumn,
    UUIDColumn,
)


def split_uuid_string_for_id(col: UUIDColumn) -> StringColumn:
    """
    Splits the UUID string into a list of strings and returns the 5th element.
    Args:
        col: The column to split
    Returns:
        The 5th element of the split UUID string
    """
    return StringColumn(F.split(col, "-")[4])


def split_last_chars(col: StringColumn) -> HexStringColumn:
    """
    Splits the last 4 characters of the column.
    Args:
        col: The column to split
    Returns:
        The last 4 characters of the column
    """
    return HexStringColumn(F.substring(col, -4, 4))


def extract_id_from_uuid(col: UUIDColumn) -> IntegerColumn:
    """
    Extracts an integer ID from a UUID4 string

    This extracts the last 4 hex characters of the UUID4 string and converts them to an integer.

    Args:
        col: The column to extract the ID from. Expected to be a UUID4 string.
    Returns:
        The integer ID from the UUID string
    """
    return convert_hex_string_to_int(split_last_chars(split_uuid_string_for_id(col)))


def convert_hex_string_to_int(col: HexStringColumn) -> IntegerColumn:
    """
    Converts a hex string to an integer

    NOTE: If the column contains invalid hex characters, the result will be 0.
    NOTE: If the column is null, the result will be null.
    NOTE: If the hex string overflows the bigint range, the result will be null.

    Args:
        col: The column to convert. Should be a string of hex characters.
    Returns:
        The bigint value of the hex string as a column
    """
    return IntegerColumn(F.conv(col, 16, 10).cast("bigint"))


def modulus_equals_offset(col: IntegerColumn, modulus: int, offset: int) -> BooleanColumn:
    """
    Checks if the modulus of the column is equal to the offset
    Args:
        col: The column to check. Expected to be an int or bigint.
        modulus: The modulus to check
        offset: The offset to check
    Returns:
        True if the modulus of the column is equal to the offset, False otherwise
    """
    return BooleanColumn(F.pmod(col, modulus) == offset)


def filter_uuid_for_modulus_and_offset(
    df: DataFrame,
    column_name: Annotated[str, "Name of UUID column"],
    modulus: Annotated[int, "Number of partitions for horizontal scaling"],
    offset: Annotated[int, "Which partition to select (0 to modulus-1)"] = 0,
) -> DataFrame:
    """
    Filters the DataFrame to only include rows where the modulus of the
    last 4 hex characters of the column is equal to the offset
    Args:
        df: The DataFrame to filter
        column_name: The name of the column to filter on. Expected to be a UUID4 string.
        modulus: The modulus to check
        offset: The offset to check
    Returns:
        The filtered DataFrame
    """
    uuid_col = UUIDColumn(F.col(column_name))
    return df.filter(
        modulus_equals_offset(
            extract_id_from_uuid(uuid_col),
            modulus,
            offset,
        )
    )
