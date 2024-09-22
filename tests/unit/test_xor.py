from __future__ import annotations

import pyspark.sql.functions as F
from pyspark_utils.xor import xor
from pyspark_utils.xor import xor_word

from tests.helpers import run_column
from tests.helpers import xor_python


def test_xor():
    """
    Test that we can XOR two strings
    """
    a = b'Hello'
    b = b'World'
    expected_result = xor_python(a, b)
    definition = xor_word(F.col('d1'), F.col('d2'))
    pyspark_result = format(run_column(definition, a, b), 'x')
    assert expected_result == pyspark_result


def test_xor_single():
    """
    Test that we can XOR two single byte strings
    """
    a = b'a'
    b = b'b'
    definition = xor_word(F.col('d1'), F.col('d2'))
    pyspark_result = run_column(definition, a, b)
    assert pyspark_result == 3


def test_xor_empty_byte_string():
    """
    Test that we can XOR two empty byte strings
    """
    a = b'\x00' * 8
    b = b'\x36' * 8
    expected_result = xor_python(a, b)
    definition = xor_word(F.col('d1'), F.col('d2'))
    pyspark_result = run_column(definition, a, b)
    assert int(expected_result, 16) == pyspark_result


def test_xor_bytes():
    """
    Test that we can XOR two byte strings
    """
    a = b'\x00a'
    b = b'\x00b'
    definition = xor_word(F.col('d1'), F.col('d2'))
    pyspark_result = run_column(definition, a, b)
    assert pyspark_result == 3


def test_xor_128():
    """
    Test that we can XOR two 16 byte strings
    """
    a = b'1111111122222222'
    b = b'4444444433333333'
    expected_result = xor_python(a, b)
    definition = xor(F.col('d1'), F.col('d2'), byte_width=16)
    pyspark_result = bytes(run_column(definition, a, b)).hex()
    assert expected_result == pyspark_result


def test_xor_odd_width_inputs():
    """
    Test that we can XOR two strings of odd length
    """
    a = b'111111112222222'
    b = b'444444443333333'
    byte_width = 16
    expected_result = xor_python(a, b, byte_width=byte_width)
    definition = xor(F.col('d1'), F.col('d2'), byte_width=byte_width)
    pyspark_result = bytes(run_column(definition, a, b)).hex()
    assert expected_result == pyspark_result


def test_xor_256():
    """
    Test that we can XOR two 32 byte strings
    """
    a = b'11111111222222223333333344444444'
    b = b'44444444333333332222222211111111'
    expected_result = xor_python(a, b)
    definition = xor(F.col('d1'), F.col('d2'), byte_width=32)
    pyspark_result = bytes(run_column(definition, a, b)).hex()
    assert expected_result == pyspark_result


def test_xor_word_overflow():
    """
    xor_word should return None if it overflows the integer conversion
    """
    a = b'\xFF' * 8
    b = b'\x36' * 8
    definition = xor_word(F.col('d1'), F.col('d2'))
    pyspark_result = run_column(definition, a, b)
    # This test confirms we will overflow on the integer conversion
    # if the value is < 2^63-1
    assert pyspark_result is None
