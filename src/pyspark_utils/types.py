from typing import NewType

from pyspark.sql import Column

BooleanColumn = NewType("BooleanColumn", Column)
ByteColumn = NewType("ByteColumn", Column)
IntegerColumn = NewType("IntegerColumn", Column)
LongColumn = NewType("LongColumn", Column)
StringColumn = NewType("StringColumn", Column)
HexStringColumn = NewType("HexStringColumn", Column)
UUIDColumn = NewType("UUIDColumn", Column)
