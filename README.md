# pyspark-toolkit

A collection of useful PySpark utility functions for data processing, including UUID generation, JSON handling, data partitioning, and cryptographic operations.

## Installation

```bash
pip install pyspark-toolkit
```

## Quick Start

```python
import pyspark.sql.functions as F
from pyspark_toolkit.uuid import uuid5
from pyspark_toolkit.json import map_json_column
from pyspark_toolkit.modulus import partition_by_uuid
from pyspark_toolkit.xor import xor
from pyspark_toolkit.helpers import map_concat

# Your PySpark code here
```

## Examples

### UUID5 Generation

Generate deterministic UUIDs from one or more columns:

```python
from pyspark_toolkit.uuid import uuid5
import uuid

# Generate UUID5 from a single column
df = spark.createDataFrame([("alice",), ("bob",)], ["name"])
df = df.withColumn("user_id", uuid5("name"))

# Generate UUID5 from multiple columns with custom separator
df = spark.createDataFrame([
    ("alice", "smith", 30),
    ("bob", "jones", 25)
], ["first", "last", "age"])
df = df.withColumn("person_id", uuid5("first", "last", "age", separator="|"))

# Use different namespace
df = df.withColumn("dns_uuid", uuid5("first", "last", namespace=uuid.NAMESPACE_DNS))

# Handle null values with custom placeholder
df = df.withColumn("uuid_nullsafe", uuid5("first", "last", null_placeholder="MISSING"))
```

### JSON Column Mapping

Parse and extract JSON data from string columns:

```python
from pyspark_toolkit.json import map_json_column, extract_json_keys_as_columns

# Parse JSON string to structured column
df = spark.createDataFrame([
    ('{"name": "Alice", "age": 30, "city": "NYC"}',),
    ('{"name": "Bob", "age": 25, "city": "LA"}',)
], ["json_data"])

# Convert JSON string to StructType
df = map_json_column(df, "json_data")

# Extract JSON keys as separate columns
df = extract_json_keys_as_columns(df, "json_data")
# Result: DataFrame with columns: json_data, name, age, city

# Keep original raw column by specifying output_column
df = map_json_column(df, "json_data", output_column="json_data_parsed")
# Result: DataFrame with both json_data (original string) and json_data_parsed (parsed)
```

### UUID-based Data Partitioning

Partition data horizontally using UUID values for distributed processing:

```python
from pyspark_toolkit.modulus import partition_by_uuid

# Create sample data with UUIDs
df = spark.createDataFrame([
    ("550e8400-e29b-41d4-a716-446655440001", "record1"),
    ("550e8400-e29b-41d4-a716-446655440002", "record2"),
    ("550e8400-e29b-41d4-a716-446655440003", "record3"),
    ("550e8400-e29b-41d4-a716-446655440004", "record4"),
], ["uuid", "data"])

# Split data into 4 partitions for parallel processing
num_partitions = 4
partitions = []
for partition_id in range(num_partitions):
    partition = partition_by_uuid(
        df,
        uuid_column="uuid",
        num_partitions=num_partitions,
        partition_id=partition_id
    )
    partitions.append(partition)

# Each partition can be processed independently
# Useful for parallel batch processing, data migration, or distributed analysis
```

### Map Concatenation

Concatenate multiple map columns with right-override merge strategy. This provides an alternative to PySpark's built-in `map_concat` function when you cannot set `spark.sql.mapKeyDedupPolicy=LAST_WIN` (e.g., in shared Databricks environments or managed clusters):

```python
from pyspark_toolkit.helpers import map_concat
import pyspark.sql.functions as F

# Create sample data with map columns
df = spark.createDataFrame([
    ({"a": 1, "b": 2}, {"c": 3, "d": 4}),
    ({"x": 10, "y": 20}, {"y": 200, "z": 30})
], ["map1", "map2"])

# Concatenate maps - rightmost values win for duplicate keys
df = df.withColumn("merged", map_concat(F.col("map1"), F.col("map2")))
# Result: {"a": 1, "b": 2, "c": 3, "d": 4} and {"x": 10, "y": 200, "z": 30}

# Concatenate multiple maps
df = spark.createDataFrame([
    ({"a": 1}, {"a": 2, "b": 3}, {"b": 4, "c": 5})
], ["map1", "map2", "map3"])

df = df.withColumn("result", map_concat(F.col("map1"), F.col("map2"), F.col("map3")))
# Result: {"a": 2, "b": 4, "c": 5} (rightmost wins: a from map2, b from map3)
```

### XOR Operations

Perform bitwise XOR operations on binary/string columns:

```python
from pyspark_toolkit.xor import xor, xor_word

# XOR two binary columns
df = spark.createDataFrame([
    (b"hello", b"world"),
    (b"foo", b"bar")
], ["col1", "col2"])

# XOR with 64-byte width (default)
df = df.withColumn("xor_result", xor(F.col("col1"), F.col("col2")))

# XOR shorter strings (max 8 chars) to get integer result
df = df.withColumn("xor_int", xor_word(F.col("col1"), F.col("col2")))

# Custom byte width
df = df.withColumn("xor_128", xor(F.col("col1"), F.col("col2"), byte_width=128))
```

### Additional JSON Processing

Examples for advanced JSON operations:

```python
from pyspark_toolkit.json import explode_all_list_columns, clean_dataframe_with_separate_lists

# Create DataFrame with nested JSON containing arrays
df = spark.createDataFrame([
    ('{"users": ["alice", "bob"], "scores": [95, 87], "active": [true, false]}',)
], ["json_col"])

# Parse JSON and explode all list columns simultaneously
df = map_json_column(df, "json_col")
df = explode_all_list_columns(df, ["users", "scores", "active"])
# Result: Each array element gets its own row with matching indices

# Clean complex nested JSON structures
df = clean_dataframe_with_separate_lists(df, "json_col")
```

### HMAC Operations

Generate HMAC-SHA256 hashes for data integrity:

```python
from pyspark_toolkit.hmac import hmac_sha256

df = spark.createDataFrame([
    ("secret_key", "message_to_hash"),
    ("another_key", "different_message")
], ["key", "message"])

df = df.withColumn("hmac", hmac_sha256(F.col("key"), F.col("message")))
```

## Available Functions

### UUID Operations
- `uuid5()` - Generate RFC 4122 compliant UUID version 5

### JSON Processing
- `map_json_column()` - Parse JSON strings to structured columns
- `extract_json_keys_as_columns()` - Extract JSON object keys as DataFrame columns
- `explode_all_list_columns()` - Explode multiple array columns with matching indices
- `explode_array_of_maps()` - Explode arrays containing map/struct objects
- `clean_dataframe_with_separate_lists()` - Clean JSON with separate array fields
- `clean_dataframe_with_single_list()` - Clean JSON with single array of objects

### Data Partitioning
- `partition_by_uuid()` - Partition data by UUID for horizontal scaling
- `extract_id_from_uuid()` - Extract integer ID from UUID for partitioning
- `modulus_equals_offset()` - Check if value matches modulus/offset criteria

### Cryptographic Operations
- `xor()` - Bitwise XOR of two binary columns
- `xor_word()` - XOR for short strings (≤8 chars) returning integer
- `hmac_sha256()` - HMAC-SHA256 hash generation

### Map Operations
- `map_concat()` - Concatenate multiple map columns with right-override merge strategy

### Utilities
- `safe_cast()` - Version-aware casting (PySpark 3.5/4.0 compatible)
- `chars_to_int()` - Convert character bytes to integer
- `pad_key()` - Pad binary key with zeros to specified block size
- `sha2_binary()` - Generate binary SHA-2 hash from input column
- `split_last_chars()` - Extract last 4 characters from string column
- `split_uuid_string_for_id()` - Extract UUID string components for partitioning

## Compatibility

- Python 3.9+
- PySpark 3.5+ (tested with 3.5.4 and 4.0)

## Known Issues

See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for information about deprecated modules.
