from __future__ import annotations

from typing import Annotated

import pyspark.sql.functions as F
from pyspark.sql import Column

from pyspark_toolkit.helpers import chars_to_int
from pyspark_toolkit.types import ByteColumn, LongColumn, StringColumn


def xor_word(
    col1: ByteColumn | StringColumn,
    col2: ByteColumn | StringColumn,
) -> LongColumn:
    """
    Tales two columns references of string data and returns the XOR of the two columns

    Max length of the string is 8 characters (xor as a 64 bit integer)

    Returns an integer representation of the bitwise XOR of the two columns
    """
    return LongColumn(chars_to_int(col1).bitwiseXOR(chars_to_int(col2)))


def xor(
    col1: Annotated[Column, "First ByteColumn to XOR"],
    col2: Annotated[Column, "Second ByteColumn to XOR"],
    byte_width: int = 64,
) -> Annotated[Column, "XOR of the two columns, as ByteColumn"]:
    """
    Takes two columns references of string data and returns the XOR of the two columns

    Max length of the string is 8 characters (xor as a 64 bit integer)

    Returns an integer representation of the bitwise XOR of the two columns

    Args:
        col1: The first column to XOR
        col2: The second column to XOR
        byte_width: The number of bytes to use for the XOR
    Returns:
        The XOR of the two columns as a ByteColumn
    """
    # Use 4 bytes (32 bits) as the word width
    # since we are XORing using 64 bit *signed* integers
    # so we cant use the full width without overflow (NULL in pyspark)
    word_width = 4
    padded_col1 = F.lpad(
        col1,
        byte_width,
        b"\x00",  # type: ignore (we need to pass bytes to rpad)
    )  # Left-pad col1 with '0' up to byte_width
    padded_col2 = F.lpad(
        col2,
        byte_width,
        b"\x00",  # type: ignore (we need to pass bytes to lpad)
    )  # Left-pad col2 with '0' up to byte_width

    chunks = []
    for i in range(0, byte_width, word_width):
        c1_chunk = F.substring(padded_col1, i + 1, word_width)
        c2_chunk = F.substring(padded_col2, i + 1, word_width)

        # XOR the two chunks
        xor_chunk = xor_word(ByteColumn(c1_chunk), ByteColumn(c2_chunk))

        # Convert XOR result to hexadecimal and pad it
        xor_hex_padded = F.lpad(
            F.hex(xor_chunk),
            2 * word_width,
            "0",
        )  # We want string 0, not byte 0 here because it is hex
        chunks.append(xor_hex_padded)

    return F.to_binary(F.concat(*chunks), F.lit("hex"))
