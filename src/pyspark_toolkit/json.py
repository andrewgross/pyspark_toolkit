from typing import Optional

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, MapType, StructType


def map_json_column(df: DataFrame, column: str, output_column: Optional[str] = None) -> DataFrame:
    """
    Takes a column of JSON strings and remaps them to a Map type by
    inferring the schema from the first row.

    Args:
        df: The DataFrame to map the JSON column
        column: The column to map
        output_column: The column to map the JSON to. If not provided, the original column is used and overwritten.
    Returns:
        The DataFrame with the JSON column mapped to a Map type
    """
    if output_column is None:
        output_column = column

    # Get first non-null JSON string to infer schema
    sample_row = df.filter(F.col(column).isNotNull()).first()
    if sample_row is None:
        raise ValueError(f"No non-null JSON strings found in column '{column}'")
    sample_json = sample_row[column]

    # Infer schema from the sample
    schema = F.schema_of_json(F.lit(sample_json))

    # Parse JSON using the inferred schema
    df = df.withColumn(output_column, F.from_json(F.col(column), schema=schema))
    return df


def extract_json_keys_as_columns(df: DataFrame, json_column: str) -> DataFrame:
    """
    Extracts each top-level key from a parsed JSON column and creates separate columns for each key.

    Args:
        df: Input PySpark DataFrame
        json_column: Name of the column containing parsed JSON (as MapType or StructType)
    Returns:
        DataFrame with top-level keys extracted as individual columns
    """
    # Get schema of the parsed JSON column
    schema = df.schema[json_column].dataType

    if not isinstance(schema, (StructType, MapType)):
        raise ValueError(f"Column '{json_column}' must be of StructType or MapType.")

    # Extract top-level keys as new columns
    if isinstance(schema, StructType):
        for field in schema.fields:
            df = df.withColumn(field.name, F.col(f"{json_column}.{field.name}"))
    elif isinstance(schema, MapType):
        keys = df.select(F.map_keys(F.col(json_column))).rdd.flatMap(lambda x: x[0]).distinct().collect()
        for key in keys:
            df = df.withColumn(key, F.col(json_column).getItem(key))
    return df


def explode_all_list_columns(df: DataFrame) -> DataFrame:
    """
    Explodes all columns in the DataFrame that are lists (ArrayType) into rows,
    ensuring they are exploded together. Adds an 'index' column representing the
    array index of the exploded value.

    ...
    "items": [1, 2, 3],
    "cost": [a, b, c],
    ...

    Args:
        df: Input PySpark DataFrame
    Returns:
        DataFrame with exploded rows and an 'index' column
    """
    list_columns = [field.name for field in df.schema.fields if isinstance(field.dataType, ArrayType)]

    if not list_columns:
        raise ValueError("No columns with ArrayType found in the DataFrame.")

    temp_col = "temp_struct"
    df = df.withColumn(temp_col, F.arrays_zip(*[F.col(col) for col in list_columns]))

    # Use select with posexplode to extract both index and values
    exploded_df = df.select(
        *[col for col in df.columns if col != temp_col],
        F.posexplode_outer(F.col(temp_col)).alias("index", "temp_struct_exploded"),
    )

    for col in list_columns:
        exploded_df = exploded_df.withColumn(col, F.col(f"temp_struct_exploded.{col}"))

    return exploded_df.drop("temp_struct", "temp_struct_exploded")


def clean_dataframe_with_separate_lists(
    df: DataFrame,
    raw_response_col: str = "raw_response",
) -> DataFrame:
    """
    Cleans up the invoice DataFrame by:
    1. Parsing the JSON column
    2. Extracting top-level keys as separate columns
    3. Exploding list columns into rows

    Args:
        df: Input PySpark DataFrame
        raw_response_col: The column containing the raw response
    Returns:
        The DataFrame with the line items exploded into separate columns
    """
    df = map_json_column(df, raw_response_col)
    df = extract_json_keys_as_columns(df, raw_response_col)
    df = explode_all_list_columns(df)
    return df


def explode_array_of_maps(df: DataFrame, array_col: str) -> DataFrame:
    """
    Explodes an array column containing maps into separate columns.

    Takes a DataFrame with an array column containing maps and explodes it into separate rows,
    with each map's keys becoming new columns. The original array column is replaced with individual
    columns for each key in the maps, suffixed with the original column name.

    ...
    items = [
        {"cost": 1.0, "other_cost": 0.12},
        {"cost": 2.0, "other_cost": 0.13},
    ]
    ...

    Args:
        df (pyspark.sql.DataFrame): Input DataFrame containing the array column
        array_col (str): Name of the array column containing maps to explode

    Returns:
        pyspark.sql.DataFrame: DataFrame with array column exploded into separate columns and rows,
            with an additional 'index' column indicating the position in the original array
    """
    df_exploded = df.select("*", F.posexplode_outer(F.col(array_col)).alias("index", "map"))

    # Extract the keys dynamically from the schema
    map_schema = df.select(F.explode(F.col(array_col)).alias("map")).schema["map"].dataType
    keys = [field.name for field in map_schema.fields]  # type: ignore (map_schema is a StructType)

    # Select original columns, map keys, and index as the last field
    original_cols = [F.col(col) for col in df.columns if col != array_col]
    return df_exploded.select(
        *original_cols,
        *[F.col(f"map.{key}").alias(f"{array_col}_{key}") for key in keys],
        F.col("index"),
    )


def clean_dataframe_with_single_list(
    df: DataFrame,
    list_col: str = "line_items",
    raw_response_col: str = "raw_response",
) -> DataFrame:
    """
    Cleans up the invoice DataFrame by:
    1. Parsing the JSON column
    2. Extracting top-level keys as separate columns
    3. Exploding line_item column into rows

    Args:
        df: Input PySpark DataFrame
        list_col: The column containing the list
        raw_response_col: The column containing the raw response
    Returns:
        The DataFrame with the line items exploded into separate columns
    """
    df = map_json_column(df, raw_response_col)
    df = extract_json_keys_as_columns(df, raw_response_col)
    df = explode_array_of_maps(df, list_col)
    return df
