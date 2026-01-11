# Simple Databricks Code - Just Save Parquet File for Download
# Add this to your Databricks notebook

from pyspark.sql import functions as F

# Your existing code
df = spark.read.parquet(
    "s3a://sw-dmi-data-staging/users/dadi.biton/test_prompt_tagging/retention_forever/prompt_tagging/trend_aggregation/"
).filter(
    F.col("category").isin(
        "Technology",
        "Financial Services",
        "Aerospace & Defense"
    )
)

# Display the data
df.display()

# ============================================
# SIMPLE SOLUTION: Just save to DBFS for download
# ============================================

# Save to DBFS (Databricks File System)
output_path = "dbfs:/FileStore/topic_analysis_data.parquet"

# Coalesce to 1 partition for single file output
df.coalesce(1).write.mode("overwrite").parquet(output_path)

print("✅ Data saved successfully!")
print(f"📁 File location: {output_path}")
print("\n📥 To download:")
print("1. Go to Databricks UI → Data → DBFS")
print("2. Navigate to: /FileStore/")
print("3. Find: topic_analysis_data.parquet")
print("4. Click Download")
print("\n📤 Then in Streamlit:")
print("1. Select 'Local File' in sidebar")
print("2. Upload the downloaded .parquet file")
print("3. Done! ✅")

# Alternative: Save with specific name matching your Streamlit app
# If you want to match the filename your Streamlit app expects:
output_path_named = "dbfs:/FileStore/Untitled_Notebook_2025_12_24_14_34_56.parquet"
df.coalesce(1).write.mode("overwrite").parquet(output_path_named)
print(f"\n✅ Also saved as: {output_path_named}")
