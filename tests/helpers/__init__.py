from pyspark.sql import SparkSession


def make_df(d1, d2):
    data = [
        (d1, d2),
    ]
    spark = SparkSession.builder.getOrCreate()
    return spark.createDataFrame(data, ["d1", "d2"])

def run_column(column_definition, d1, d2):
    df = make_df(d1, d2)
    df = df.withColumn("result", column_definition)
    return df.collect()[0]["result"]

def xor_python(d1: str, d2: str):
    b1 = bytes(d1, "utf-8")
    b2 = bytes(d2, "utf-8")
    result =  bytearray([a^b for a, b in zip(b1, b2)])
    return result.hex()
