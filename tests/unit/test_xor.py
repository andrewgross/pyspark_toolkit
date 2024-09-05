from pyspark_utils.xor import xor_pyspark
from tests.helpers import xor_python


def test_xor():
    a = "Hello"
    b = "World"
    expected_result = xor_python(a, b)
    pyspark_result = xor_pyspark(a, b)
    assert expected_result == pyspark_result