from typing import Union

import pyspark.sql.functions as F
import pyspark.sql.types as T

from pyspark_utils.helpers import (
    HexStringColumn,
    LongColumn,
    StringColumn,
    string_to_int,
)


def xor_word(col1: Union[str, F.Column], col2: Union[str, F.Column]) -> LongColumn:
    """
    Tales two columns references of string data and returns the XOR of the two columns

    Max length of the string is 8 characters (xor as a 64 bit integer)

    Returns an integer representation of the bitwise XOR of the two columns
    """
    return string_to_int(col1).bitwiseXOR(string_to_int(col2))


def xor(
    col1: StringColumn, col2: StringColumn, bit_width: int = 512
) -> HexStringColumn:

    max_len = bit_width // 8  # 8 bits in a byte, each character is 1 byte
    padded_col1 = F.lpad(col1, max_len, "0")  # Left-pad col1 with '0' up to max_len
    padded_col2 = F.lpad(col2, max_len, "0")  # Left-pad col2 with '0' up to max_len

    chunks = []
    for i in range(0, max_len, 8):
        c1_chunk = F.substring(padded_col1, i + 1, 8)
        c2_chunk = F.substring(padded_col2, i + 1, 8)

        # XOR the two chunks
        xor_chunk = xor_word(c1_chunk, c2_chunk)

        # Convert XOR result to hexadecimal and pad it
        xor_hex_padded = F.lpad(
            F.hex(xor_chunk), 16, "0"
        )  # We want string 0, not byte 0 here because it is hex
        chunks.append(xor_hex_padded)

    return F.concat(*chunks)
