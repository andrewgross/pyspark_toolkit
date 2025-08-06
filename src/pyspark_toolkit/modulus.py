from typing import Annotated

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from pyspark_toolkit.helpers import split_last_chars
from pyspark_toolkit.types import BooleanColumn, IntegerColumn, StringColumn, UUIDColumn


def split_uuid_string_for_id(col: UUIDColumn) -> StringColumn:
    """
    Splits the UUID string into a list of strings and returns the 5th element.
    Args:
        col: The column to split
    Returns:
        The 5th element of the split UUID string
    """
    return StringColumn(F.split(col, "-")[4])


def extract_id_from_uuid(col: UUIDColumn) -> IntegerColumn:
    """
    Extracts an integer ID from a UUID4 string

    This extracts the last 4 hex characters of the UUID4 string and converts them to an integer.

    Args:
        col: The column to extract the ID from. Expected to be a UUID4 string.
    Returns:
        The integer ID from the UUID string
    """
    hex_chars = split_last_chars(split_uuid_string_for_id(col))
    # Convert hex string directly to integer (not via chars_to_int which is for byte conversion)
    return IntegerColumn(F.conv(hex_chars, 16, 10).cast("bigint"))


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


def partition_by_uuid(
    df: DataFrame,
    uuid_column: Annotated[str, "Name of UUID column"],
    num_partitions: Annotated[int, "Total number of partitions"],
    partition_id: Annotated[int, "Which partition to select (0 to num_partitions-1)"] = 0,
) -> DataFrame:
    """
    Partition DataFrame by UUID for horizontal scaling.

    Uses the last 4 hex characters of a UUID to deterministically assign rows
    to partitions. Useful for distributing data processing across multiple
    workers or systems.

    Args:
        df: The DataFrame to partition
        uuid_column: The name of the UUID column to partition by
        num_partitions: Total number of partitions to split data into
        partition_id: Which partition to return (0 to num_partitions-1)

    Returns:
        DataFrame containing only rows belonging to the specified partition

    Example:
        >>> # Split data into 4 partitions for parallel processing
        >>> for i in range(4):
        ...     partition = partition_by_uuid(df, "uuid", 4, i)
        ...     process_partition(partition)
    """
    uuid_col = UUIDColumn(F.col(uuid_column))
    return df.filter(
        modulus_equals_offset(
            extract_id_from_uuid(uuid_col),
            num_partitions,
            partition_id,
        )
    )
