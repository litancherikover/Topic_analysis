# Databricks Notebook - Download Filtered CSV to Desktop
# Add this code to your Databricks notebook

from pyspark.sql import functions as F

# Filter the data
df_filtered = spark.read.parquet(
    "s3a://sw-dmi-data-staging/users/dadi.biton/test_prompt_tagging/retention_forever/prompt_tagging/trend_aggregation/"
).filter(
    F.col("category").isin(
        "Technology",
        "Financial Services",
        "Aerospace & Defense"
    )
)

# Display the filtered data
df_filtered.display()

# ============================================
# Save as CSV for download
# ============================================

# Save to DBFS as CSV (single file)
output_path = "dbfs:/FileStore/Prompts.csv"

# Coalesce to 1 partition and save as CSV with header
df_filtered.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

print("=" * 60)
print("✅ Filtered data saved as Prompts.csv!")
print("=" * 60)
print(f"📁 Location: {output_path}")
print("\n📥 HOW TO DOWNLOAD TO YOUR DESKTOP:")
print("-" * 60)
print("1. Go to Databricks UI → Data → DBFS")
print("2. Navigate to: /FileStore/Prompts.csv/")
print("3. Find the .csv file inside (e.g., part-00000-xxx.csv)")
print("4. Download it and rename to Prompts.csv")
print("=" * 60)

# Show file info
print(f"\n📊 Number of rows: {df_filtered.count():,}")
