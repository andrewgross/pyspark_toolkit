from typing import Union

from pyspark.sql import functions as F
from pyspark.sql import types as T


def string_to_int(col: Union[str, F.Column]) -> "F.Column[T.LongType()]":
    """
    Take a string, encode it as utf-8, and convert those bytes to a bigint

    Currently blows up if our string is too big
    """
    return F.conv(F.hex(F.to_binary(col, F.lit("utf-8"))), 16, 10).cast("bigint")