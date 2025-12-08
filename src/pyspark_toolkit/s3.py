from __future__ import annotations

import datetime
from typing import Union

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.column import Column

from pyspark_toolkit.hmac import hmac_sha256
from pyspark_toolkit.types import ByteColumn

try:
    from pyspark.sql.connect.column import Column as ConnectColumn
except ImportError:
    ConnectColumn = Column  # fallback for typing only

_COLUMN = Union[Column, ConnectColumn]

# Prefix for all intermediate columns to avoid collisions
_TEMP_COL_PREFIX = "__s3_presign_"


def _resolve_to_column(df: DataFrame, value: Union[str, int, _COLUMN]) -> _COLUMN:
    """
    Convert a value to a Column, handling strings, literals, and Column objects.

    Resolution order:
    1. If already a Column (F.col, F.lit, or expression), use directly
    2. If string matching an existing column name, treat as column reference
    3. Otherwise, treat as literal value
    """
    if isinstance(value, _COLUMN):
        return value
    elif isinstance(value, str) and value in df.columns:
        return F.col(value)
    else:
        return F.lit(value)


def _to_binary(col):
    """Convert a string column to binary using UTF-8 encoding."""
    return F.encode(col, "UTF-8")


def generate_presigned_url(
    df: DataFrame,
    bucket: Union[str, Column],
    key: Union[str, Column],
    aws_access_key: Union[str, Column],
    aws_secret_key: Union[str, Column],
    region: Union[str, Column],
    expiration: Union[str, int, Column],
    output_col: str = "presigned_url",
) -> DataFrame:
    """
    Generate presigned URLs for S3 objects using AWS Signature Version 4.

    This function adds a new column containing presigned URLs for S3 GET requests.
    The computation is staged across multiple intermediate columns to avoid
    deep expression trees that can cause Spark's Catalyst optimizer to OOM.

    Each parameter (except df and output_col) can be provided as:
    - A string matching an existing column name (treated as column reference)
    - A string not matching any column (treated as literal value)
    - An integer (treated as literal value, for expiration)
    - A Column object (F.col("name"), F.lit("value"), or any column expression)

    Args:
        df: Input DataFrame.
        bucket: S3 bucket name - column reference or literal value.
        key: S3 object key (path) - column reference or literal value.
        aws_access_key: AWS access key - column reference or literal value.
        aws_secret_key: AWS secret key - column reference or literal value.
        region: AWS region name - column reference or literal value.
        expiration: Expiration time in seconds - column reference or literal value.
        output_col: Name of the output column for presigned URLs.

    Returns:
        DataFrame with the presigned URL column added.

    Example:
        >>> # Using column references for bucket/key, literals for credentials
        >>> df = spark.createDataFrame([
        ...     ("my-bucket", "path/to/file.txt")
        ... ], ["bucket", "key"])
        >>> result = generate_presigned_url(
        ...     df, "bucket", "key",
        ...     aws_access_key="AKIAIOSFODNN7EXAMPLE",
        ...     aws_secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        ...     region="us-east-1",
        ...     expiration=3600
        ... )
    """
    # Define temp column names
    t = lambda name: f"{_TEMP_COL_PREFIX}{name}"

    # Resolve all inputs to Column objects and store in temp columns
    # This avoids repeating the resolution logic and keeps expressions simple
    df = df.withColumns(
        {
            t("bucket"): _resolve_to_column(df, bucket),
            t("key"): _resolve_to_column(df, key),
            t("aws_access_key"): _resolve_to_column(df, aws_access_key),
            t("aws_secret_key"): _resolve_to_column(df, aws_secret_key),
            t("region"): _resolve_to_column(df, region),
            t("expiration"): _resolve_to_column(df, expiration),
        }
    )
    # Checkpoint the DataFrame to avoid repeated evaluation of the same expressions
    # when using F.lit() values it explodes the expression tree and hangs execution
    df = df.localCheckpoint(eager=False)

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
    df = df.withColumn(t("canonical_uri"), F.concat(F.lit("/"), F.col(t("key"))))
    df = df.withColumn(
        t("host"),
        F.concat(F.col(t("bucket")), F.lit(".s3."), F.col(t("region")), F.lit(".amazonaws.com")),
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
            F.col(t("aws_access_key")),
            F.lit("%2F"),
            F.col(t("date_stamp")),
            F.lit("%2F"),
            F.col(t("region")),
            F.lit("%2Fs3%2Faws4_request"),
            F.lit("&X-Amz-Date="),
            F.col(t("amz_date")),
            F.lit("&X-Amz-Expires="),
            F.col(t("expiration")).cast("string"),
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
            F.col(t("region")),
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
        _to_binary(F.concat(F.lit("AWS4"), F.col(t("aws_secret_key")))),
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
        _to_binary(F.col(t("region"))),
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
