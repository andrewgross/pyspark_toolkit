from __future__ import annotations

import json

import pyspark.sql.functions as F
import pyspark.sql.types as T
import pytest

from pyspark_utils.json_functions import (
    clean_dataframe_with_separate_line_item_lists,
    clean_dataframe_with_single_line_item_list,
    explode_all_list_columns,
    explode_array_of_maps,
    extract_json_keys_as_columns,
    map_json_column,
)


def test_map_json_column(spark):
    """
    Test that map_json_column correctly parses JSON strings into StructType.
    """
    # when I have a dataframe with JSON strings
    data = [
        ('{"name": "Alice", "age": 30, "city": "NYC"}',),
        ('{"name": "Bob", "age": 25, "city": "LA"}',),
        ('{"name": "Charlie", "age": 35, "city": "Chicago"}',),
    ]
    df = spark.createDataFrame(data, ["json_data"])

    # and I apply map_json_column
    result_df = map_json_column(df, "json_data")

    # then I should get a dataframe with parsed JSON as StructType
    assert "json_data" in result_df.columns
    assert "json_data_raw" not in result_df.columns

    schema = result_df.schema["json_data"].dataType
    assert isinstance(schema, T.StructType)
    assert len(schema.fields) == 3
    assert set(f.name for f in schema.fields) == {"name", "age", "city"}

    row = result_df.first()
    assert row.json_data.name == "Alice"
    assert row.json_data.age == 30
    assert row.json_data.city == "NYC"


def test_map_json_column_with_drop_false(spark):
    """
    Test that map_json_column preserves raw column when drop=False.
    """
    # when I have a dataframe with JSON strings
    data = [
        ('{"key": "value1"}',),
        ('{"key": "value2"}',),
    ]
    df = spark.createDataFrame(data, ["json_data"])

    # and I apply map_json_column with drop=False
    result_df = map_json_column(df, "json_data", drop=False)

    # then I should get both the parsed JSON and the raw column
    assert "json_data" in result_df.columns
    assert "json_data_raw" in result_df.columns

    row = result_df.first()
    assert row.json_data.key == "value1"
    assert row.json_data_raw == '{"key": "value1"}'


def test_extract_json_keys_as_columns_struct_type(spark):
    """
    Test that extract_json_keys_as_columns extracts keys from StructType columns.
    """
    # when I have a dataframe with parsed JSON as StructType
    data = [
        ('{"name": "Alice", "age": 30, "active": true}',),
        ('{"name": "Bob", "age": 25, "active": false}',),
    ]
    df = spark.createDataFrame(data, ["json_data"])
    df = map_json_column(df, "json_data")

    # and I apply extract_json_keys_as_columns
    result_df = extract_json_keys_as_columns(df, "json_data")

    # then I should get separate columns for each key
    assert "name" in result_df.columns
    assert "age" in result_df.columns
    assert "active" in result_df.columns

    row = result_df.first()
    assert row.name == "Alice"
    assert row.age == 30
    assert row.active is True


def test_extract_json_keys_as_columns_map_type(spark):
    """
    Test that extract_json_keys_as_columns extracts keys from MapType columns.
    """
    # when I have a dataframe with MapType column
    schema = T.StructType([T.StructField("json_data", T.MapType(T.StringType(), T.StringType()))])
    data = [
        ({"name": "Alice", "age": "30"},),
        ({"name": "Bob", "age": "25"},),
    ]
    df = spark.createDataFrame(data, schema)

    # and I apply extract_json_keys_as_columns
    result_df = extract_json_keys_as_columns(df, "json_data")

    # then I should get separate columns for each key
    assert "name" in result_df.columns
    assert "age" in result_df.columns

    row = result_df.first()
    assert row["name"] == "Alice"
    assert row["age"] == "30"


def test_extract_json_keys_as_columns_invalid_type(spark):
    """
    Test that extract_json_keys_as_columns raises ValueError for invalid column types.
    """
    # when I have a dataframe with non-JSON column
    data = [(1,), (2,)]
    df = spark.createDataFrame(data, ["number"])

    # and I apply extract_json_keys_as_columns
    # then I should get a ValueError
    with pytest.raises(ValueError) as excinfo:
        extract_json_keys_as_columns(df, "number")

    assert "must be of StructType or MapType" in str(excinfo.value)


def test_explode_all_list_columns(spark):
    """
    Test that explode_all_list_columns explodes multiple arrays together with indices.
    """
    # when I have a dataframe with multiple list columns
    data = [
        ("order1", [1, 2, 3], ["item1", "item2", "item3"], [10.0, 20.0, 30.0]),
        ("order2", [4, 5], ["item4", "item5"], [40.0, 50.0]),
    ]
    df = spark.createDataFrame(data, ["order_id", "quantities", "items", "prices"])

    # and I apply explode_all_list_columns
    result_df = explode_all_list_columns(df)

    # then I should get rows exploded with matching indices
    assert "index" in result_df.columns

    rows = result_df.orderBy("order_id", "index").collect()
    assert len(rows) == 5

    assert rows[0].order_id == "order1"
    assert rows[0]["index"] == 0
    assert rows[0].quantities == 1
    assert rows[0].items == "item1"
    assert rows[0].prices == 10.0

    assert rows[2].order_id == "order1"
    assert rows[2]["index"] == 2
    assert rows[2].quantities == 3
    assert rows[2].items == "item3"
    assert rows[2].prices == 30.0


def test_explode_all_list_columns_no_arrays(spark):
    """
    Test that explode_all_list_columns raises ValueError when no array columns exist.
    """
    # when I have a dataframe with no list columns
    data = [
        ("order1", 1, "item1"),
        ("order2", 2, "item2"),
    ]
    df = spark.createDataFrame(data, ["order_id", "quantity", "item"])

    # and I apply explode_all_list_columns
    # then I should get a ValueError
    with pytest.raises(ValueError) as excinfo:
        explode_all_list_columns(df)

    assert "No columns with ArrayType found" in str(excinfo.value)


def test_explode_array_of_maps(spark):
    """
    Test that explode_array_of_maps correctly explodes arrays containing maps.
    """
    # when I have a dataframe with an array of maps column
    data = [
        (
            "order1",
            [
                {"item": "apple", "cost": 1.0, "quantity": 2},
                {"item": "banana", "cost": 0.5, "quantity": 3},
            ],
        ),
        (
            "order2",
            [
                {"item": "orange", "cost": 0.75, "quantity": 1},
            ],
        ),
    ]

    schema = T.StructType(
        [
            T.StructField("order_id", T.StringType()),
            T.StructField(
                "line_items",
                T.ArrayType(
                    T.StructType(
                        [
                            T.StructField("item", T.StringType()),
                            T.StructField("cost", T.DoubleType()),
                            T.StructField("quantity", T.IntegerType()),
                        ]
                    )
                ),
            ),
        ]
    )

    df = spark.createDataFrame(data, schema)

    # and I apply explode_array_of_maps
    result_df = explode_array_of_maps(df, "line_items")

    # then I should get separate rows with map keys as columns
    assert "line_items_item" in result_df.columns
    assert "line_items_cost" in result_df.columns
    assert "line_items_quantity" in result_df.columns
    assert "index" in result_df.columns
    assert "line_items" not in result_df.columns

    rows = result_df.orderBy("order_id", "index").collect()
    assert len(rows) == 3

    assert rows[0].order_id == "order1"
    assert rows[0].line_items_item == "apple"
    assert rows[0].line_items_cost == 1.0
    assert rows[0].line_items_quantity == 2
    assert rows[0]["index"] == 0

    assert rows[1].order_id == "order1"
    assert rows[1].line_items_item == "banana"
    assert rows[1]["index"] == 1


def test_clean_dataframe_with_separate_line_item_lists(spark):
    """
    Test end-to-end cleaning of JSON with separate array fields.
    """
    # when I have a dataframe with JSON containing separate list fields
    json_data = {
        "customer": "John Doe",
        "total": 100.0,
        "items": ["item1", "item2", "item3"],
        "quantities": [1, 2, 3],
        "prices": [10.0, 20.0, 30.0],
    }

    data = [
        ("invoice1", json.dumps(json_data)),
    ]
    df = spark.createDataFrame(data, ["invoice_id", "raw_response"])

    # and I apply clean_dataframe_with_separate_line_item_lists
    result_df = clean_dataframe_with_separate_line_item_lists(df)

    # then I should get a cleaned dataframe with exploded lists
    assert "customer" in result_df.columns
    assert "total" in result_df.columns
    assert "items" in result_df.columns
    assert "quantities" in result_df.columns
    assert "prices" in result_df.columns
    assert "index" in result_df.columns

    rows = result_df.orderBy("index").collect()
    assert len(rows) == 3

    assert rows[0].customer == "John Doe"
    assert rows[0].total == 100.0
    assert rows[0].items == "item1"
    assert rows[0].quantities == 1
    assert rows[0].prices == 10.0


def test_clean_dataframe_with_single_line_item_list(spark):
    """
    Test end-to-end cleaning of JSON with a single array of maps.
    """
    # when I have a dataframe with JSON containing a single line_items array
    json_data = {
        "customer": "Jane Smith",
        "total": 75.0,
        "line_items": [
            {"item": "apple", "cost": 25.0, "quantity": 1},
            {"item": "banana", "cost": 50.0, "quantity": 2},
        ],
    }

    data = [
        ("invoice2", json.dumps(json_data)),
    ]
    df = spark.createDataFrame(data, ["invoice_id", "raw_response"])

    # and I apply clean_dataframe_with_single_line_item_list
    result_df = clean_dataframe_with_single_line_item_list(df)

    # then I should get a cleaned dataframe with exploded line items
    assert "customer" in result_df.columns
    assert "total" in result_df.columns
    assert "line_items_item" in result_df.columns
    assert "line_items_cost" in result_df.columns
    assert "line_items_quantity" in result_df.columns
    assert "index" in result_df.columns
    assert "line_items" not in result_df.columns

    rows = result_df.orderBy("index").collect()
    assert len(rows) == 2

    assert rows[0].customer == "Jane Smith"
    assert rows[0].total == 75.0
    assert rows[0].line_items_item == "apple"
    assert rows[0].line_items_cost == 25.0
    assert rows[0].line_items_quantity == 1


def test_map_json_column_with_nulls(spark):
    """
    Test that map_json_column handles null values gracefully.
    """
    # when I have a dataframe with JSON strings including nulls
    data = [
        ('{"name": "Alice", "age": 30}',),
        (None,),
        ('{"name": "Bob", "age": 25}',),
    ]
    df = spark.createDataFrame(data, ["json_data"])

    # and I apply map_json_column
    result_df = map_json_column(df, "json_data")

    # then nulls should be handled gracefully
    rows = result_df.collect()
    assert rows[0].json_data.name == "Alice"
    assert rows[1].json_data is None
    assert rows[2].json_data.name == "Bob"


def test_explode_all_list_columns_with_different_lengths(spark):
    """
    Test that explode_all_list_columns handles arrays of different lengths.
    """
    # when I have a dataframe with list columns of different lengths
    data = [
        ("order1", [1, 2, 3], ["item1", "item2"]),
    ]

    schema = T.StructType(
        [
            T.StructField("order_id", T.StringType()),
            T.StructField("quantities", T.ArrayType(T.IntegerType())),
            T.StructField("items", T.ArrayType(T.StringType())),
        ]
    )

    df = spark.createDataFrame(data, schema)

    # and I apply explode_all_list_columns
    result_df = explode_all_list_columns(df)

    # then I should get nulls for shorter lists
    rows = result_df.orderBy("index").collect()
    assert len(rows) == 3

    assert rows[2].quantities == 3
    assert rows[2].items is None
