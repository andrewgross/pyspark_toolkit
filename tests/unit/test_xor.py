import pyspark.sql.functions as F

from pyspark_utils.xor import xor, xor_word
from tests.helpers import run_column, xor_python


def test_xor():
    a = "Hello"
    b = "World"
    expected_result = xor_python(a, b)
    definition = xor_word(F.col("d1"), F.col("d2"))
    pyspark_result = format(run_column(definition, a, b), "x")
    assert expected_result == pyspark_result


def test_xor_single():
    a = "a"
    b = "b"
    definition = xor_word(F.col("d1"), F.col("d2"))
    pyspark_result = run_column(definition, a, b)
    assert pyspark_result == 3


def test_xor_128():
    a = "1111111122222222"
    b = "4444444433333333"
    expected_result = xor_python(a, b)
    definition = xor(F.col("d1"), F.col("d2"), bit_width=128)
    pyspark_result = run_column(definition, a, b)
    assert expected_result == pyspark_result


def test_xor_256():
    a = "11111111222222223333333344444444"
    b = "44444444333333332222222211111111"
    expected_result = xor_python(a, b)
    definition = xor(F.col("d1"), F.col("d2"), bit_width=256)
    pyspark_result = run_column(definition, a, b)
    assert expected_result == pyspark_result
