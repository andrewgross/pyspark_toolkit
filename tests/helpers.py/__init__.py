from pyspark.sql import SparkSession


def make_df(d1, d2):
    spark = SparkSession.builder.getOrCreate()
    data = [
        (d1, d2),
    ]
    return spark.createDataFrame(data, ["d1", "d2"])

def xor_python(d1: str, d2: str):
    b1 = bytes(d1, "utf-8")
    b2 = bytes(d2, "utf-8")
    result =  bytearray([a^b for a, b in zip(b1, b2)])
    return int.from_bytes(result, byteorder='big', signed=False)