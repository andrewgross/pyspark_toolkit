from __future__ import annotations

import warnings

import pytest


def test_s3_module_import_warning():
    """
    Test that importing the s3 module raises a deprecation warning.
    """
    with pytest.warns(DeprecationWarning, match="s3 module is deprecated"):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            import pyspark_utils.s3  # noqa: F401


def test_s3_excluded_from_package():
    """
    Test that s3 module raises deprecation warning when imported.

    This documents that the s3 module should be excluded from the built package.
    The actual hanging test is in test_s3_isolated.py and runs via process isolation.
    """
    # When running from source, the module exists but should warn
    # When installed from package, it shouldn't exist at all
    with pytest.warns(DeprecationWarning, match="s3 module is deprecated"):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            # Force reimport to trigger warning
            import importlib
            import sys

            if "pyspark_utils.s3" in sys.modules:
                del sys.modules["pyspark_utils.s3"]
            import pyspark_utils.s3  # noqa: F401


def test_generate_presigned_url_warns():
    """
    Test that generate_presigned_url raises a warning.

    This verifies our warning mechanism is working.
    Note: The actual hanging behavior is tested in test_s3_isolated.py
    """
    # Import inside test to avoid module-level warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import pyspark.sql.functions as F

        from pyspark_utils.s3 import generate_presigned_url

    # The function should warn when called
    with pytest.warns(RuntimeWarning, match="non-functional due to deep call graph"):
        # Just calling the function should trigger the warning
        # We don't evaluate it to avoid the hang
        result = generate_presigned_url(
            F.col("bucket"),
            F.col("key"),
            F.col("access_key"),
            F.col("secret_key"),
            F.col("region"),
            F.col("expiration"),
        )
        # Result is a Column expression, not evaluated yet
        assert result is not None
