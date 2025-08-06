# Known Issues

## S3 Module - Non-functional

**Status:** Deprecated/Excluded from package distribution

**Issue:** The `s3.py` module contains code for generating AWS S3 presigned URLs using PySpark UDFs. However, this implementation suffers from deep call graph issues when using HMAC functions that cause server hangs during execution.

**Impact:** The repeated nested calls to `hmac_sha256` in the signature generation process create a call graph that's too deep for PySpark to handle efficiently, leading to server hangs.

**Testing:**
- Run `make test-s3-isolated` to verify the hanging issue (uses process isolation with 5s timeout)
- The test will timeout as expected, confirming the issue still exists
- If the test ever completes successfully, the issue may be resolved in your PySpark version

**Workaround:**
- The module is excluded from the distributed package to prevent accidental usage
- The code remains in the repository for reference and potential future fixes
- Users should generate S3 presigned URLs outside of PySpark UDFs using standard AWS SDKs

**Files Affected:**
- `src/pyspark_utils/s3.py` - Contains the problematic implementation with warnings
- `tests/unit/s3_isolated_runner.py` - Isolated test that demonstrates the hanging issue
- `tests/run_s3_timeout_test.py` - Test runner with process isolation and timeout
- `pyproject.toml` - Configured to exclude the s3 module from distribution

**Note:** The code includes runtime warnings if imported directly from source. Do not use in production environments.
