"""Demo script to show cdtf output."""

from pyspark.sql import SparkSession

from pyspark_toolkit.udtf import cdtf

spark = SparkSession.builder.appName("cdtf_demo").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Create sample data
df = spark.createDataFrame(
    [(1, "apple"), (2, "banana"), (3, "cherry"), (4, "date")],
    ["id", "fruit"],
)

print("=== Input DataFrame ===")
df.show()


# Demo with retries - some succeed after retries, some fail completely
print("\n=== Demo with Retries ===")


def retry_init(self):
    self.attempt_counts = {}  # track attempts per row


@cdtf(
    output_schema="result STRING",
    init_fn=retry_init,
    max_workers=2,
    max_retries=3,  # allow up to 3 retries (4 total attempts)
)
def flaky_process(self, row):
    row_id = row["id"]

    # Track attempts for this row
    if row_id not in self.attempt_counts:
        self.attempt_counts[row_id] = 0
    self.attempt_counts[row_id] += 1
    attempt = self.attempt_counts[row_id]

    # id=1: succeeds first try
    # id=2: fails twice, succeeds on 3rd attempt
    # id=3: fails all 4 attempts
    # id=4: succeeds first try

    if row_id == 2 and attempt < 3:
        raise ValueError(f"Temporary failure for id=2 (attempt {attempt})")

    if row_id == 3:
        raise ValueError(f"Permanent failure for id=3 (attempt {attempt})")

    return (f"success_{row['fruit']}",)


result_df = flaky_process(df)

print("Results:")
result_df.show(truncate=False)

print("\nSchema:")
result_df.printSchema()

print("\nDetailed view of each row:")
for row in sorted(result_df.collect(), key=lambda r: r["id"]):
    print(f"\n  id={row['id']} ({row['fruit']}):")
    print(f"    result: {row['result']}")
    print(f"    _metadata ({len(row['_metadata'])} attempts):")
    for attempt in row["_metadata"]:
        status = "SUCCESS" if attempt["error"] is None else f"FAILED: {attempt['error']}"
        print(f"      attempt {attempt['attempt']}: {status} ({attempt['duration_ms']}ms)")

spark.stop()
