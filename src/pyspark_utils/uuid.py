import uuid

import pyspark.sql.functions as F
from pyspark.sql import Column


def uuid5(
    *columns: str | Column,
    namespace=uuid.NAMESPACE_OID,
    separator: str = "-",
    null_placeholder: str = "\0",
) -> Column:
    """
    Generates a UUIDv5 from the provided columns and namespace, using a custom separator (default "-").

    This function creates a RFC 4122/9562 compliant UUIDv5 string using PySpark. It concatenates the input columns
    with the specified separator, then uses this concatenated string along with the provided namespace to generate the UUID.

    Currently no support for complex types (arrays, structs, maps, variants).

    Args:
        *columns: A list of string column names or Column objects. If strings are provided, they are converted to Column objects.
        namespace: The namespace to use for the UUID generation. Defaults to uuid.NAMESPACE_OID.
        separator: The separator to use when concatenating columns. Defaults to "-".
        null_placeholder: The placeholder to use for null values. Defaults to "\0" (null byte).
    """
    if not columns:
        raise ValueError("No columns passed!")

    normalized_columns = []
    for col in columns:
        if isinstance(col, str):
            col = F.col(col)
        normalized_col = F.coalesce(col, F.lit(null_placeholder))
        normalized_columns.append(normalized_col)

    name = F.concat_ws(separator, *normalized_columns)
    name_bytes = F.encode(name, "UTF-8")
    sha1_hash = F.sha1(F.concat(F.lit(namespace.bytes), name_bytes))

    # Format UUID components
    return F.concat_ws(
        "-",
        F.substring(sha1_hash, 1, 8),  # Time Low
        F.substring(sha1_hash, 9, 4),  # Time Mid
        F.concat(F.lit("5"), F.substring(sha1_hash, 14, 3)),  # Time High and Version
        F.concat(
            F.lower(  # We use lower here because F.hex returns Alphabetical characters as capitals
                F.concat(
                    F.hex(
                        # Take the 17th "char", convert to bytes and ensure the Most Significant bit is high
                        # And reconvert to hex to merge with the rest of the substring
                        F.conv(F.substring(sha1_hash, 17, 1), 16, 10)
                        .cast("int")
                        .bitwiseAND(F.lit(0x3))
                        .bitwiseOR(F.lit(0x8))
                    ),
                    F.substring(sha1_hash, 18, 3),
                )
            )
        ),  # Variant
        F.substring(sha1_hash, 21, 12),  # Rest of String
    )
