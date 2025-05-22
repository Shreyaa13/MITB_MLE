import os
from datetime import datetime
import pyspark.sql.functions as F

def process_feature_store_gold_table(snapshot_date_str, silver_features_dir, gold_feature_store_dir, spark):
    
    # Paths to silver tables for the given snapshot_date
    silver_attributes_path = os.path.join(silver_features_dir, "attributes", f"silver_attributes_{snapshot_date_str.replace('-', '_')}.parquet")
    silver_financials_path = os.path.join(silver_features_dir, "financials", f"silver_financials_{snapshot_date_str.replace('-', '_')}.parquet")
    silver_clickstream_path = os.path.join(silver_features_dir, "clickstream", f"silver_clickstream_{snapshot_date_str.replace('-', '_')}.parquet")

    df_attributes = spark.read.parquet(silver_attributes_path)
    df_financials = spark.read.parquet(silver_financials_path)
    df_clickstream = spark.read.parquet(silver_clickstream_path)

    print(f"Loaded silver_attributes for {snapshot_date_str}, count: {df_attributes.count()}")
    print(f"Loaded silver_financials for {snapshot_date_str}, count: {df_financials.count()}")
    print(f"Loaded silver_clickstream for {snapshot_date_str}, count: {df_clickstream.count()}")

    # Join attributes and financials on Customer_ID and snapshot_date
    join_condition = ["Customer_ID", "snapshot_date"]
    df_attr_fin = df_attributes.join(df_financials, join_condition, "inner")

    # Rename clickstream snapshot_date to avoid column collision
    df_clickstream = df_clickstream.withColumnRenamed("snapshot_date", "clickstream_snapshot_date")

    # Join the above with clickstream only on Customer_ID
    df_gold = df_clickstream.join(df_attr_fin, "Customer_ID", "left")

    # Add feature store snapshot_date column for clarity (from attributes/financials join)
    df_gold = df_gold.withColumn("feature_snapshot_date", F.col("snapshot_date"))

    print(f'Gold feature store for {snapshot_date_str} - joined count: {df_gold.count()}')

    # select columns to drop
    df_gold = df_gold.drop("snapshot_date", "clickstream_snapshot_date", "Credit_History_Age", 
                             "Total_EMI_per_month", "Outstanding_Debt", "Credit_Mix",
                            "Num_Credit_Card")
    
    # save gold table
    if not os.path.exists(gold_feature_store_dir):
        os.makedirs(gold_feature_store_dir)
        
    partition_name = f"gold_feature_store_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(gold_feature_store_dir, partition_name)
    df_gold.write.mode("overwrite").parquet(filepath)
    
    print(f'Saved gold feature store to: {filepath}, row count: {df_gold.count()}')
    
    return df_gold