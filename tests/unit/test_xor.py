import pyspark.sql.functions as F

from pyspark_utils.xor import xor
from tests.helpers import run_column, xor_python


def test_xor():
    a = "Hello"
    b = "World"
    expected_result = xor_python(a, b)
    definition = xor(F.col("d1"), F.col("d2"))
    pyspark_result = run_column(definition, a, b)
    assert expected_result == pyspark_result

def test_xor_single():
    a = "a"
    b = "b"
    expected_result = xor_python(a, b)
    definition = xor(F.col("d1"), F.col("d2"))
    pyspark_result = run_column(definition, a, b)
    assert expected_result == pyspark_result