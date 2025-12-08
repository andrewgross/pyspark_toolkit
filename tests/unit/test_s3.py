from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

from botocore.auth import S3SigV4QueryAuth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from freezegun import freeze_time

from pyspark_toolkit.s3 import generate_presigned_url


def test_generate_presigned_url_returns_dataframe_with_output_column(spark):
    """
    Test that generate_presigned_url adds the presigned_url column to the DataFrame.
    """
    # when I have a DataFrame with the required input columns
    df = spark.createDataFrame(
        [
            (
                "my-bucket",
                "path/to/file.txt",
                "AKIAIOSFODNN7EXAMPLE",
                "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                "us-east-1",
                3600,
            )
        ],
        ["bucket", "key", "access_key", "secret_key", "region", "expiration"],
    )

    # and I generate a presigned URL
    result = generate_presigned_url(
        df,
        bucket="bucket",
        key="key",
        aws_access_key="access_key",
        aws_secret_key="secret_key",
        region="region",
        expiration="expiration",
    )

    # then the result should have the presigned_url column
    assert "presigned_url" in result.columns

    # and no intermediate columns should remain
    temp_cols = [c for c in result.columns if c.startswith("__s3_presign_")]
    assert len(temp_cols) == 0


def test_generate_presigned_url_custom_output_column(spark):
    """
    Test that generate_presigned_url allows custom output column name.
    """
    # when I have a DataFrame with the required input columns
    df = spark.createDataFrame(
        [
            (
                "my-bucket",
                "file.txt",
                "AKIAIOSFODNN7EXAMPLE",
                "secret",
                "us-west-2",
                7200,
            )
        ],
        ["bucket", "key", "access_key", "secret_key", "region", "expiration"],
    )

    # and I generate a presigned URL with a custom output column name
    result = generate_presigned_url(
        df,
        bucket="bucket",
        key="key",
        aws_access_key="access_key",
        aws_secret_key="secret_key",
        region="region",
        expiration="expiration",
        output_col="my_url",
    )

    # then the result should have the custom column name
    assert "my_url" in result.columns
    assert "presigned_url" not in result.columns


def test_generate_presigned_url_format_is_valid(spark):
    """
    Test that the generated presigned URL has the correct format.
    """
    # when I have a DataFrame with the required input columns
    df = spark.createDataFrame(
        [
            (
                "test-bucket",
                "some/object/key.json",
                "AKIAIOSFODNN7EXAMPLE",
                "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                "eu-west-1",
                3600,
            )
        ],
        ["bucket", "key", "access_key", "secret_key", "region", "expiration"],
    )

    # and I generate a presigned URL
    result = generate_presigned_url(
        df,
        bucket="bucket",
        key="key",
        aws_access_key="access_key",
        aws_secret_key="secret_key",
        region="region",
        expiration="expiration",
    )

    # then the URL should have the correct format
    url = result.collect()[0]["presigned_url"]

    # URL should start with the correct endpoint
    assert url.startswith("https://test-bucket.s3.eu-west-1.amazonaws.com/some/object/key.json?")

    # URL should contain required query parameters
    assert "X-Amz-Algorithm=AWS4-HMAC-SHA256" in url
    assert "X-Amz-Credential=AKIAIOSFODNN7EXAMPLE" in url
    assert "X-Amz-Expires=3600" in url
    assert "X-Amz-SignedHeaders=host" in url
    assert "&X-Amz-Signature=" in url

    # Signature should be lowercase hex
    signature_match = re.search(r"X-Amz-Signature=([a-f0-9]+)$", url)
    assert signature_match is not None
    assert len(signature_match.group(1)) == 64  # SHA-256 produces 64 hex chars


def test_generate_presigned_url_preserves_original_columns(spark):
    """
    Test that generate_presigned_url preserves all original columns.
    """
    # when I have a DataFrame with extra columns beyond the required ones
    df = spark.createDataFrame(
        [
            (
                "bucket",
                "key.txt",
                "access",
                "secret",
                "us-east-1",
                3600,
                "extra_value",
                123,
            )
        ],
        [
            "bucket",
            "key",
            "access_key",
            "secret_key",
            "region",
            "expiration",
            "extra_col",
            "another_col",
        ],
    )

    # and I generate a presigned URL
    result = generate_presigned_url(
        df,
        bucket="bucket",
        key="key",
        aws_access_key="access_key",
        aws_secret_key="secret_key",
        region="region",
        expiration="expiration",
    )

    # then all original columns should be preserved
    assert "extra_col" in result.columns
    assert "another_col" in result.columns

    row = result.collect()[0]
    assert row["extra_col"] == "extra_value"
    assert row["another_col"] == 123


def test_generate_presigned_url_handles_multiple_rows(spark):
    """
    Test that generate_presigned_url works correctly with multiple rows.
    """
    # when I have a DataFrame with multiple rows
    df = spark.createDataFrame(
        [
            ("bucket1", "key1.txt", "access1", "secret1", "us-east-1", 3600),
            ("bucket2", "key2.txt", "access2", "secret2", "us-west-2", 7200),
            ("bucket3", "key3.txt", "access3", "secret3", "eu-west-1", 1800),
        ],
        ["bucket", "key", "access_key", "secret_key", "region", "expiration"],
    )

    # and I generate presigned URLs
    result = generate_presigned_url(
        df,
        bucket="bucket",
        key="key",
        aws_access_key="access_key",
        aws_secret_key="secret_key",
        region="region",
        expiration="expiration",
    )

    # then each row should have a unique presigned URL
    rows = result.collect()
    urls = [row["presigned_url"] for row in rows]

    assert len(urls) == 3
    assert urls[0].startswith("https://bucket1.s3.us-east-1.amazonaws.com/key1.txt?")
    assert urls[1].startswith("https://bucket2.s3.us-west-2.amazonaws.com/key2.txt?")
    assert urls[2].startswith("https://bucket3.s3.eu-west-1.amazonaws.com/key3.txt?")

    # All URLs should be unique (different signatures due to different inputs)
    assert len(set(urls)) == 3


def _generate_boto3_presigned_url(
    bucket: str,
    key: str,
    access_key: str,
    secret_key: str,
    region: str,
    expiration: int,
) -> str:
    """
    Generate a presigned URL using botocore's S3SigV4QueryAuth.

    This uses the same signing logic as boto3's generate_presigned_url for S3.
    S3SigV4QueryAuth uses UNSIGNED-PAYLOAD and does not normalize URL paths,
    which matches our PySpark implementation.
    The timestamp should be controlled externally using freezegun.
    """
    credentials = Credentials(access_key, secret_key)
    auth = S3SigV4QueryAuth(credentials, "s3", region, expires=expiration)

    # Build the URL that would be signed
    url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
    request = AWSRequest(method="GET", url=url)

    # Add the auth to the request (this modifies the URL with query params)
    auth.add_auth(request)

    return request.url


@freeze_time("2024-06-20 08:15:00", tz_offset=0)
def test_generate_presigned_url_matches_boto3_signature(spark):
    """
    Test that our PySpark implementation produces the same signature as boto3/botocore.

    We use freezegun to freeze time for boto3, and pass a timestamp column to our
    PySpark implementation since Spark's JVM time cannot be frozen by freezegun.
    """
    # given test credentials (AWS example credentials from documentation)
    access_key = "AKIAIOSFODNN7EXAMPLE"
    secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    bucket = "examplebucket"
    key = "test/object.txt"
    region = "us-east-1"
    expiration = 3600

    # when I generate a presigned URL using boto3 (with frozen time)
    boto3_url = _generate_boto3_presigned_url(
        bucket=bucket,
        key=key,
        access_key=access_key,
        secret_key=secret_key,
        region=region,
        expiration=expiration,
    )

    # and I generate a presigned URL using our PySpark implementation
    df = spark.createDataFrame(
        [(bucket, key, access_key, secret_key, region, expiration)],
        ["bucket", "key", "access_key", "secret_key", "region", "expiration"],
    )

    result = generate_presigned_url(
        df,
        bucket="bucket",
        key="key",
        aws_access_key="access_key",
        aws_secret_key="secret_key",
        region="region",
        expiration="expiration",
    )

    pyspark_url = result.collect()[0]["presigned_url"]

    # then both URLs should have the same signature
    boto3_parsed = urlparse(boto3_url)
    pyspark_parsed = urlparse(pyspark_url)

    boto3_params = parse_qs(boto3_parsed.query)
    pyspark_params = parse_qs(pyspark_parsed.query)

    # Verify host and path match
    assert boto3_parsed.netloc == pyspark_parsed.netloc
    assert boto3_parsed.path == pyspark_parsed.path

    # Verify key signing parameters match
    assert boto3_params["X-Amz-Algorithm"] == pyspark_params["X-Amz-Algorithm"]
    assert boto3_params["X-Amz-Credential"] == pyspark_params["X-Amz-Credential"]
    assert boto3_params["X-Amz-Date"] == pyspark_params["X-Amz-Date"]
    assert boto3_params["X-Amz-Expires"] == pyspark_params["X-Amz-Expires"]
    assert boto3_params["X-Amz-SignedHeaders"] == pyspark_params["X-Amz-SignedHeaders"]

    # Most importantly: the signatures should match
    assert boto3_params["X-Amz-Signature"] == pyspark_params["X-Amz-Signature"], (
        f"Signatures differ:\n"
        f"  boto3:   {boto3_params['X-Amz-Signature']}\n"
        f"  pyspark: {pyspark_params['X-Amz-Signature']}"
    )


@freeze_time("2024-06-20 08:15:00", tz_offset=0)
def test_generate_presigned_url_with_python_string_values(spark):
    """
    Test that generate_presigned_url works with python string values.

    When a string doesn't match any column name, it should be treated as a literal value.
    """
    # when I have a DataFrame with only bucket and key columns
    df = spark.createDataFrame(
        [
            ("my-bucket", "path/to/file.txt"),
        ],
        ["bucket", "key"],
    )

    # and I generate presigned URLs with literal values for credentials/region/expiration
    result = generate_presigned_url(
        df,
        bucket="bucket",
        key="key",
        aws_access_key="AKIAIOSFODNN7EXAMPLE",
        aws_secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        region="us-east-1",
        expiration=3600,
    )

    # then the URL should be generated successfully
    rows = result.collect()
    assert len(rows) == 1

    url = rows[0]["presigned_url"]
    assert url.startswith("https://my-bucket.s3.us-east-1.amazonaws.com/")
    assert "X-Amz-Algorithm=AWS4-HMAC-SHA256" in url
    assert "X-Amz-Credential=AKIAIOSFODNN7EXAMPLE" in url
    assert "X-Amz-Expires=3600" in url
    assert "&X-Amz-Signature=" in url


@freeze_time("2024-06-20 08:15:00", tz_offset=0)
def test_generate_presigned_url_with_f_lit_values(spark):
    """
    Test that generate_presigned_url works with F.lit() Column objects.
    """
    from pyspark.sql import functions as F

    # when I have a DataFrame with only bucket and key columns
    df = spark.createDataFrame(
        [
            ("my-bucket", "path/to/file.txt"),
        ],
        ["bucket", "key"],
    )

    # and I generate presigned URLs with F.lit() for some values
    result = generate_presigned_url(
        df,
        bucket="bucket",
        key="key",
        aws_access_key=F.lit("AKIAIOSFODNN7EXAMPLE"),
        aws_secret_key=F.lit("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
        region=F.lit("us-east-1"),
        expiration=F.lit(3600),
    )

    # then the URL should be generated correctly
    url = result.collect()[0]["presigned_url"]
    assert url.startswith("https://my-bucket.s3.us-east-1.amazonaws.com/")
    assert "X-Amz-Credential=AKIAIOSFODNN7EXAMPLE" in url
    assert "X-Amz-Expires=3600" in url


@freeze_time("2024-06-20 08:15:00", tz_offset=0)
def test_generate_presigned_url_with_f_col_values(spark):
    """
    Test that generate_presigned_url works with F.col() Column objects.
    """
    from pyspark.sql import functions as F

    # when I have a DataFrame with all values as columns
    df = spark.createDataFrame(
        [
            (
                "my-bucket",
                "path/to/file.txt",
                "AKIAIOSFODNN7EXAMPLE",
                "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                "us-east-1",
                3600,
            ),
        ],
        ["bucket", "key", "access_key", "secret_key", "region", "expiration"],
    )

    # and I generate presigned URLs with explicit F.col() references
    result = generate_presigned_url(
        df,
        bucket=F.col("bucket"),
        key=F.col("key"),
        aws_access_key=F.col("access_key"),
        aws_secret_key=F.col("secret_key"),
        region=F.col("region"),
        expiration=F.col("expiration"),
    )

    # then the URL should be generated correctly
    url = result.collect()[0]["presigned_url"]
    assert url.startswith("https://my-bucket.s3.us-east-1.amazonaws.com/")
    assert "X-Amz-Credential=AKIAIOSFODNN7EXAMPLE" in url
    assert "X-Amz-Expires=3600" in url


@freeze_time("2024-06-20 08:15:00", tz_offset=0)
def test_generate_presigned_url_with_mixed_input_types(spark):
    """
    Test that generate_presigned_url works with a mix of input types.

    This tests the common use case where bucket/key come from columns
    but credentials are literal values.
    """
    from pyspark.sql import functions as F

    # when I have a DataFrame with bucket and key columns
    df = spark.createDataFrame(
        [
            ("bucket-1", "file1.txt"),
            ("bucket-2", "file2.txt"),
        ],
        ["bucket", "key"],
    )

    # and I generate presigned URLs mixing column refs, literals, and F.lit
    result = generate_presigned_url(
        df,
        bucket="bucket",  # column name (string matching column)
        key=F.col("key"),  # explicit F.col
        aws_access_key="AKIAIOSFODNN7EXAMPLE",  # literal (no column match)
        aws_secret_key=F.lit("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),  # F.lit
        region="us-east-1",  # literal (no column match)
        expiration=3600,  # integer literal
    )

    # then URLs should be generated for each row
    rows = result.collect()
    assert len(rows) == 2

    # and they should use the correct bucket from each row
    assert rows[0]["presigned_url"].startswith("https://bucket-1.s3.us-east-1.amazonaws.com/file1.txt?")
    assert rows[1]["presigned_url"].startswith("https://bucket-2.s3.us-east-1.amazonaws.com/file2.txt?")


@freeze_time("2024-06-20 08:15:00", tz_offset=0)
def test_generate_presigned_url_matches_boto3_with_special_characters(spark):
    """
    Test that our implementation handles special characters in keys correctly.
    """
    access_key = "AKIAIOSFODNN7EXAMPLE"
    secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    bucket = "test-bucket"
    key = "path/to/file with spaces.txt"
    region = "eu-west-1"
    expiration = 7200

    # when I generate URLs using both implementations
    boto3_url = _generate_boto3_presigned_url(
        bucket=bucket,
        key=key,
        access_key=access_key,
        secret_key=secret_key,
        region=region,
        expiration=expiration,
    )

    df = spark.createDataFrame(
        [(bucket, key, access_key, secret_key, region, expiration)],
        ["bucket", "key", "access_key", "secret_key", "region", "expiration"],
    )

    result = generate_presigned_url(
        df,
        bucket="bucket",
        key="key",
        aws_access_key="access_key",
        aws_secret_key="secret_key",
        region="region",
        expiration="expiration",
    )

    pyspark_url = result.collect()[0]["presigned_url"]

    # then the signatures should match
    boto3_params = parse_qs(urlparse(boto3_url).query)
    pyspark_params = parse_qs(urlparse(pyspark_url).query)

    assert boto3_params["X-Amz-Signature"] == pyspark_params["X-Amz-Signature"], (
        f"Signatures differ for key with special characters:\n"
        f"  boto3:   {boto3_params['X-Amz-Signature']}\n"
        f"  pyspark: {pyspark_params['X-Amz-Signature']}"
    )
