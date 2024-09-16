import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession

from pyspark_utils.helpers import chars_to_int, pad_key, sha2_binary
from tests.helpers import run_column


def test_chars_to_int():
    """
    chars_to_int should convert a string to an integer
    """
    a = "Hello"
    # int.from_bytes(a.encode("utf-8"), "big")
    definition = chars_to_int(F.col("d1"))
    pyspark_result = run_column(definition, a, "")
    assert pyspark_result == 310939249775


def test_chars_to_int_with_bytes():
    """
    chars_to_int should convert a bytes object to an integer
    """
    a = b"Hello"
    # int.from_bytes(a.encode("utf-8"), "big")
    definition = chars_to_int(F.col("d1"))
    pyspark_result = run_column(definition, a, "")
    assert pyspark_result == 310939249775


def test_chars_to_int_with_unprintable_characters():
    """
    chars_to_int should convert a bytes object to an integer
    """
    a = b"\x00\x00\x00\x00\x00\x00\x00\x00"
    definition = chars_to_int(F.col("d1"))
    pyspark_result = run_column(definition, a, "")
    assert pyspark_result == 0


def test_chars_to_int_hex_bytes_overflow():
    """
    chars_to_int should return None if the input is too large
    """
    a = b"\xFF" * 8
    definition = chars_to_int(F.col("d1"))
    pyspark_result = run_column(definition, a, "")
    assert pyspark_result == None


def test_string_to_int_types():
    """
    string_to_int should return an integer
    """
    a = "Hello"
    data = [(a)]
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(data, T.StringType())
    df = df.withColumn("result", chars_to_int(F.col("value")))
    assert dict(df.dtypes)["result"] == "bigint"


def test_sha2_binary():
    """
    Test that we can hash a string and convert that to a binary
    """
    a = "Hello"
    definition = sha2_binary(F.col("d1"), 256)
    pyspark_result = run_column(definition, a, "")
    assert (
        pyspark_result
        == b'\x18_\x8d\xb3"q\xfe%\xf5a\xa6\xfc\x93\x8b.&C\x06\xec0N\xdaQ\x80\x07\xd1vH&8\x19i'
    )


def test_pad_key_equal():
    """
    Padding a key to the same length should return the same key
    """
    key = b"e179017a-62b0-4996-8a38-e91aa9f1e179017a-62b0-4996-8aaaaaaaaaaaa"
    assert len(key) == 64
    block_size = 64
    data = [
        (key),
    ]
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(data, T.BinaryType())
    df = df.withColumn("key", pad_key(F.col("value"), block_size=block_size))
    expected_result = key
    assert expected_result == bytes(df.collect()[0]["key"])


def test_pad_key_short():
    """
    Padding a key to a shorter length should add padding
    """
    key = b"1234"
    block_size = 5
    data = [
        (key),
    ]
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(data, T.BinaryType())
    df = df.withColumn("key", pad_key(F.col("value"), block_size=block_size))
    expected_result = b"1234\x00"
    assert expected_result == bytes(df.collect()[0]["key"])
