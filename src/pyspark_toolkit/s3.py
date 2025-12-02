from __future__ import annotations

import datetime

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pyspark_toolkit.hmac import hmac_sha256
from pyspark_toolkit.types import ByteColumn

# Prefix for all intermediate columns to avoid collisions
_TEMP_COL_PREFIX = "__s3_presign_"


def _to_binary(col):
    """Convert a string column to binary using UTF-8 encoding."""
    return F.encode(col, "UTF-8")


def generate_presigned_url(
    df: DataFrame,
    bucket_col: str,
    key_col: str,
    aws_access_key_col: str,
    aws_secret_key_col: str,
    region_col: str,
    expiration_col: str,
    output_col: str = "presigned_url",
) -> DataFrame:
    """
    Generate presigned URLs for S3 objects using AWS Signature Version 4.

    This function adds a new column containing presigned URLs for S3 GET requests.
    The computation is staged across multiple intermediate columns to avoid
    deep expression trees that can cause Spark's Catalyst optimizer to OOM.

    Args:
        df: Input DataFrame containing the required columns.
        bucket_col: Name of column containing S3 bucket names.
        key_col: Name of column containing S3 object keys (paths).
        aws_access_key_col: Name of column containing AWS access keys.
        aws_secret_key_col: Name of column containing AWS secret keys (string, will be encoded).
        region_col: Name of column containing AWS region names.
        expiration_col: Name of column containing expiration times in seconds.
        output_col: Name of the output column for presigned URLs.

    Returns:
        DataFrame with the presigned URL column added.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("my-bucket", "path/to/file.txt", "AKIAIOSFODNN7EXAMPLE",
        ...      "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "us-east-1", 3600)
        ... ], ["bucket", "key", "access_key", "secret_key", "region", "expiration"])
        >>> result = generate_presigned_url(
        ...     df, "bucket", "key", "access_key", "secret_key", "region", "expiration"
        ... )
    """
    # Define temp column names
    t = lambda name: f"{_TEMP_COL_PREFIX}{name}"

    # Stage 1: Timestamp values

    # We mimic the behavior of botocore.auth.get_current_datetime()
    # https://github.com/boto/botocore/blob/cee3e3141b3d5171c6ee0a0bc9bd9abcd209e594/botocore/compat.py#L309
    datetime_now = datetime.datetime.now(datetime.timezone.utc)
    datetime_now = datetime_now.replace(tzinfo=None)
    df = df.withColumn(
        t("now_utc"),
        F.lit(datetime_now),
    )
    # Format the UTC timestamp for AWS signing
    df = df.withColumn(
        t("amz_date"),
        F.concat(
            F.date_format(F.col(t("now_utc")), "yyyyMMdd"),
            F.lit("T"),
            F.date_format(F.col(t("now_utc")), "HHmmss"),
            F.lit("Z"),
        ),
    )
    df = df.withColumn(t("date_stamp"), F.date_format(F.col(t("now_utc")), "yyyyMMdd"))

    # Stage 2: URI and host components
    df = df.withColumn(t("canonical_uri"), F.concat(F.lit("/"), F.col(key_col)))
    df = df.withColumn(
        t("host"),
        F.concat(F.col(bucket_col), F.lit(".s3."), F.col(region_col), F.lit(".amazonaws.com")),
    )
    df = df.withColumn(
        t("endpoint"),
        F.concat(F.lit("https://"), F.col(t("host")), F.col(t("canonical_uri"))),
    )

    # Stage 3: Canonical query string
    # Note: The credential value must use URL-encoded slashes (%2F) as per AWS Signature v4
    df = df.withColumn(
        t("canonical_qs"),
        F.concat(
            F.lit("X-Amz-Algorithm=AWS4-HMAC-SHA256"),
            F.lit("&X-Amz-Credential="),
            F.col(aws_access_key_col),
            F.lit("%2F"),
            F.col(t("date_stamp")),
            F.lit("%2F"),
            F.col(region_col),
            F.lit("%2Fs3%2Faws4_request"),
            F.lit("&X-Amz-Date="),
            F.col(t("amz_date")),
            F.lit("&X-Amz-Expires="),
            F.col(expiration_col).cast("string"),
            F.lit("&X-Amz-SignedHeaders=host"),
        ),
    )

    # Stage 4: Canonical headers
    df = df.withColumn(
        t("canonical_headers"),
        F.concat(F.lit("host:"), F.col(t("host")), F.lit("\n")),
    )

    # Stage 5: Canonical request
    df = df.withColumn(
        t("canonical_request"),
        F.concat_ws(
            "\n",
            F.lit("GET"),
            F.col(t("canonical_uri")),  # path, possible we are not handling this correctly with quoting
            F.col(t("canonical_qs")),  # canonical_query_string
            F.col(
                t("canonical_headers")
            ),  # Do we need an extra \n here? I think its already included in the canonical_headers column
            F.lit("host"),  # signed_headers
            F.lit("UNSIGNED-PAYLOAD"),  # payload_hash,
            # We use UNSIGNED-PAYLOAD because we don't know the payload when creating the presigned URL, following the AWS documentation
        ),
    )

    # Stage 6: Credential scope and string to sign
    df = df.withColumn(
        t("credential_scope"),
        F.concat_ws(
            "/",
            F.col(t("date_stamp")),
            F.col(region_col),
            F.lit("s3"),
            F.lit("aws4_request"),
        ),
    )

    df = df.withColumn(
        t("canonical_request_hash"),
        F.sha2(F.col(t("canonical_request")), 256),
    )

    df = df.withColumn(
        t("string_to_sign"),
        F.concat_ws(
            "\n",
            F.lit("AWS4-HMAC-SHA256"),
            F.col(t("amz_date")),
            F.col(t("credential_scope")),
            F.col(t("canonical_request_hash")),
        ),
    )

    # Stage 7: Signing key derivation (staged HMAC chain)
    # All inputs to HMAC must be binary - encode strings as UTF-8
    df = df.withColumn(
        t("key_prefix"),
        _to_binary(F.concat(F.lit("AWS4"), F.col(aws_secret_key_col))),
    )

    df = df.withColumn(
        t("date_stamp_bin"),
        _to_binary(F.col(t("date_stamp"))),
    )

    df = df.withColumn(
        t("k_date"),
        hmac_sha256(
            ByteColumn(F.col(t("key_prefix"))),
            ByteColumn(F.col(t("date_stamp_bin"))),
        ),
    )

    df = df.withColumn(
        t("region_bin"),
        _to_binary(F.col(region_col)),
    )

    df = df.withColumn(
        t("k_region"),
        hmac_sha256(
            ByteColumn(F.col(t("k_date"))),
            ByteColumn(F.col(t("region_bin"))),
        ),
    )

    df = df.withColumn(
        t("k_service"),
        hmac_sha256(
            ByteColumn(F.col(t("k_region"))),
            ByteColumn(F.lit(b"s3")),
        ),
    )

    df = df.withColumn(
        t("signing_key"),
        hmac_sha256(
            ByteColumn(F.col(t("k_service"))),
            ByteColumn(F.lit(b"aws4_request")),
        ),
    )

    # Stage 8: Final signature
    df = df.withColumn(
        t("string_to_sign_bin"),
        _to_binary(F.col(t("string_to_sign"))),
    )

    df = df.withColumn(
        t("signature"),
        F.hex(
            hmac_sha256(
                ByteColumn(F.col(t("signing_key"))),
                ByteColumn(F.col(t("string_to_sign_bin"))),
            )
        ),
    )

    # Stage 9: Assemble final URL
    df = df.withColumn(
        output_col,
        F.concat(
            F.col(t("endpoint")),
            F.lit("?"),
            F.col(t("canonical_qs")),
            F.lit("&X-Amz-Signature="),
            F.lower(F.col(t("signature"))),
        ),
    )

    # Clean up intermediate columns
    temp_cols = [c for c in df.columns if c.startswith(_TEMP_COL_PREFIX)]
    df = df.drop(*temp_cols)

    return df
