import hashlib
import hmac
from itertools import zip_longest
from typing import Union

from pyspark.sql import SparkSession


def make_df(d1, d2):
    data = [
        (d1, d2),
    ]
    spark = SparkSession.builder.getOrCreate()
    return spark.createDataFrame(data, ["d1", "d2"])


def run_column(column_definition, d1, d2):
    df = make_df(d1, d2)
    df = df.withColumn("result", column_definition)
    return df.collect()[0]["result"]


def xor_bytes_pad(a: bytes, b: bytes, pad: int = 0) -> bytes:
    return bytes(x ^ y for x, y in zip_longest(a, b, fillvalue=pad))


def xor_python(d1: bytes, d2: bytes, byte_width=None):
    result = xor_bytes_pad(d1, d2)
    hex_result = result.hex()
    if byte_width is not None:
        return hex_result.zfill(byte_width * 2)
    return hex_result


def hmac_python(
    key: Union[str, bytes], message: Union[str, bytes], digest=hashlib.sha256
) -> str:
    if isinstance(key, str):
        b1 = bytes(key, "utf-8")
    else:
        b1 = key
    if isinstance(message, str):
        b2 = bytes(message, "utf-8")
    else:
        b2 = message
    return hmac.new(key, message, digest).hexdigest()
