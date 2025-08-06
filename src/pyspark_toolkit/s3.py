from __future__ import annotations

import warnings

from pyspark.sql import functions as F

from pyspark_toolkit.hmac import hmac_sha256
from pyspark_toolkit.types import ByteColumn, IntegerColumn, StringColumn

warnings.warn(
    "The s3 module is deprecated and non-functional due to deep call graph issues with HMAC "
    "that cause server hangs. Do not use this module in production.",
    DeprecationWarning,
    stacklevel=2,
)


def generate_presigned_url(
    bucket: StringColumn,
    key: StringColumn,
    aws_access_key: ByteColumn,
    aws_secret_key: ByteColumn,
    region: StringColumn,
    expiration: IntegerColumn,
) -> StringColumn:
    """
    Generate a presigned URL for an S3 object using AWS Signature Version 4.

    **WARNING: This function is non-functional due to deep call graph issues with HMAC
    that cause server hangs. Do not use in production.**

    Args:
        bucket (StringColumn): The name of the S3 bucket.
        key (StringColumn): The key (path) of the S3 object.
        aws_access_key (ByteColumn): The AWS access key.
        aws_secret_key (ByteColumn): The AWS secret key.
        region (StringColumn): The AWS region where the S3 bucket is located.
        expiration (IntegerColumn): The expiration time in seconds for the presigned URL.

    Returns:
        StringColumn: The presigned URL for the S3 object.
    """
    warnings.warn(
        "generate_presigned_url is non-functional due to deep call graph issues with HMAC "
        "that cause server hangs. Do not use in production.",
        RuntimeWarning,
        stacklevel=2,
    )

    # Step 1: Get the current UTC timestamp
    now = F.current_timestamp()

    # Step 2: Generate formatted date strings (amz_date and date_stamp)
    amz_date = F.date_format(now, "yyyyMMdd'T'HHmmss'Z'")
    date_stamp = F.date_format(now, "yyyyMMdd")

    # Step 3: Create the canonical URI and host
    canonical_uri = F.concat(F.lit("/"), F.lit(key))
    host = F.concat(bucket, F.lit(".s3."), region, F.lit(".amazonaws.com"))
    endpoint = F.concat(F.lit("https://"), host, canonical_uri)

    # Step 4: Build the canonical query string
    canonical_querystring = F.concat(
        F.lit("X-Amz-Algorithm=AWS4-HMAC-SHA256"),
        F.lit("&X-Amz-Credential="),
        F.lit(aws_access_key),
        F.lit("/"),
        date_stamp,
        F.lit("/"),
        F.lit(region),
        F.lit("/s3/aws4_request"),
        F.lit("&X-Amz-Date="),
        amz_date,
        F.lit("&X-Amz-Expires="),
        F.lit(str(expiration)),
        F.lit("&X-Amz-SignedHeaders=host"),
    )

    # Step 5: Create the canonical headers and signed headers
    canonical_headers = F.concat(F.lit("host:"), host, F.lit("\n"))
    signed_headers = F.lit("host")

    # Step 6: Set payload hash (we'll use 'UNSIGNED-PAYLOAD' as per S3 requirements for GET requests)
    payload_hash = F.lit("UNSIGNED-PAYLOAD")

    # Step 7: Build the canonical request
    canonical_request = F.concat(
        F.lit("GET\n"),
        canonical_uri,
        F.lit("\n"),
        canonical_querystring,
        F.lit("\n"),
        canonical_headers,
        F.lit("\n"),
        signed_headers,
        F.lit("\n"),
        payload_hash,
    )

    # Step 8: Create the string to sign
    credential_scope = F.concat(
        date_stamp,
        F.lit("/"),
        region,
        F.lit("/s3/aws4_request"),
    )

    string_to_sign = F.concat(
        F.lit("AWS4-HMAC-SHA256\n"),
        amz_date,
        F.lit("\n"),
        credential_scope,
        F.lit("\n"),
        F.sha2(canonical_request, 256),
    )

    # Step 9: Generate the signing key
    signing_key = _get_signature_key(aws_secret_key, date_stamp, region, "s3")

    # Step 10: Call hmac_256 function to generate the signature
    signature = hmac_sha256(ByteColumn(signing_key), ByteColumn(string_to_sign))

    # Step 11: Build the final signed URL
    signed_url = F.concat(
        endpoint,
        F.lit("?"),
        canonical_querystring,
        F.lit("&X-Amz-Signature="),
        signature,
    )

    return StringColumn(signed_url)


def _get_signature_key(aws_secret_key, date_stamp, region, service):
    """
    Helper function for AWS signature generation.

    **WARNING: This function is non-functional due to deep call graph issues.**
    """
    key_prefix = F.concat(
        F.lit("AWS4"),
        aws_secret_key,
    )
    k_date = hmac_sha256(ByteColumn(key_prefix), ByteColumn(date_stamp))
    k_region = hmac_sha256(k_date, region)
    k_service = hmac_sha256(k_region, service)
    signing_key = hmac_sha256(ByteColumn(k_service), ByteColumn(F.lit("aws4_request")))
    return signing_key
