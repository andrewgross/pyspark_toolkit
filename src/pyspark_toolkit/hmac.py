from __future__ import annotations

import pyspark.sql.functions as F

from pyspark_toolkit.helpers import pad_key, sha2_binary
from pyspark_toolkit.types import ByteColumn
from pyspark_toolkit.xor import xor


def hmac_sha256(key: ByteColumn, message: ByteColumn) -> ByteColumn:
    """
    Compute the HMAC-SHA256 of a message using a key

    :param key: The key to use for the HMAC. This should be a Column of bytes, not a string column name
    :param message: The message to hash. This should be a Column of bytes, not a string column name
    """
    block_size = 64

    prepared_key = _prepare_key(key, block_size)

    # Create the inner and outer padding
    i_key_pad = xor(prepared_key, F.lit(b"\x36" * block_size))
    o_key_pad = xor(prepared_key, F.lit(b"\x5c" * block_size))

    # Perform inner hash
    inner_hash = sha2_binary(ByteColumn(F.concat(i_key_pad, message)), 256)

    # Perform outer hash
    hmac = sha2_binary(ByteColumn(F.concat(o_key_pad, inner_hash)), 256)
    return hmac


def _prepare_key(key: ByteColumn, block_size: int, digest: int = 256) -> ByteColumn:
    """
    Prepare the key for HMAC by hashing it if it is longer than the block size

    Then pad the key to the block size
    """
    # The output size of a SHA-256 hash is 32 bytes
    hashed_key = sha2_binary(key, digest)
    key_within_block_size = F.greatest(F.length(key), F.lit(block_size)).eqNullSafe(
        F.lit(block_size),
    )
    key_source = F.when(key_within_block_size, key).otherwise(hashed_key)
    return pad_key(ByteColumn(key_source), block_size)
