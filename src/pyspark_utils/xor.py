from typing import Union

from pyspark.sql import functions as F
from pyspark.sql import types as T

from pyspark_utils.helpers import string_to_int


def xor(col1: Union[str, F.Column], col2: Union[str, F.Column]) -> "F.Column[T.LongType()]":
    """
    Currently takes two columns (string or reference) of string/bytes

    Runs the Xor

    Returns the value as an int

    Should it instead return bytes again? String? Should it handle int inputs?
    """
    return string_to_int(col1).bitwiseXOR(string_to_int(col2))

def xor_256(col1: Union[str, F.Column], col2: Union[str, F.Column]) -> "F.Column[T.LongType()]":
    """
    Currently takes two columns (string or reference) of string/bytes

    Runs the Xor

    Returns the value as an int

    Should it instead return bytes again? String? Should it handle int inputs?
    """
    return string_to_int(col1).bitwiseXOR(string_to_int(col2))