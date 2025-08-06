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

# Keep original raw column
df = map_json_column(df, "json_data", drop=False)
# Result: DataFrame with both json_data (parsed) and json_data_raw (original string)
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

### Utilities
- `safe_cast()` - Version-aware casting (PySpark 3.5/4.0 compatible)
- `chars_to_int()` - Convert character bytes to integer

## Compatibility

- Python 3.9+
- PySpark 3.5+ (tested with 3.5.4 and 4.0)

## Known Issues

See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for information about deprecated modules.
