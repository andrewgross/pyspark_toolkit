import pyspark.sql.functions as F

from pyspark_utils.xor import xor, xor_word
from tests.helpers import run_column, xor_python


def test_xor():
    a = b"Hello"
    b = b"World"
    expected_result = xor_python(a, b)
    definition = xor_word(F.col("d1"), F.col("d2"))
    pyspark_result = format(run_column(definition, a, b), "x")
    assert expected_result == pyspark_result


def test_xor_single():
    a = b"a"
    b = b"b"
    definition = xor_word(F.col("d1"), F.col("d2"))
    pyspark_result = run_column(definition, a, b)
    assert pyspark_result == 3


def test_xor_bytes():
    a = b"\x00a"
    b = b"\x00b"
    definition = xor_word(F.col("d1"), F.col("d2"))
    pyspark_result = run_column(definition, a, b)
    assert pyspark_result == 3


def test_xor_128():
    a = b"1111111122222222"
    b = b"4444444433333333"
    expected_result = xor_python(a, b)
    definition = xor(F.col("d1"), F.col("d2"), byte_width=16)
    pyspark_result = bytes(run_column(definition, a, b)).hex()
    assert expected_result == pyspark_result


def test_xor_odd_width_inputs():
    a = b"111111112222222"
    b = b"444444443333333"
    byte_width = 16
    expected_result = xor_python(a, b, byte_width=byte_width)
    definition = xor(F.col("d1"), F.col("d2"), byte_width=byte_width)
    pyspark_result = bytes(run_column(definition, a, b)).hex()
    assert expected_result == pyspark_result


def test_xor_256():
    a = b"11111111222222223333333344444444"
    b = b"44444444333333332222222211111111"
    expected_result = xor_python(a, b)
    definition = xor(F.col("d1"), F.col("d2"), byte_width=32)
    pyspark_result = bytes(run_column(definition, a, b)).hex()
    assert expected_result == pyspark_result


def test_xor_hashed_key():
    # Gotta fix this test, something is going sideways in our XOR
    block_size = 64
    a = b"\xd1|\xa7\x95\x17\xb1\xe2\x18\x99\xb6\xe9\xda<O/3@\xaf\xe8\xae\xc0\r\xdb\x1f\x1a#\xa9\xd6\x9e\xd6f\xed\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b = b"\x36" * block_size
    expected_result = xor_python(a, b)
    definition = xor(F.col("d1"), F.col("d2"), byte_width=block_size)
    pyspark_result = bytes(run_column(definition, a, b)).hex()
    assert expected_result == pyspark_result
