import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

# Common function to process a generic feature CSV to bronze
def process_feature_to_bronze(snapshot_date_str, bronze_base_dir, feature_name, raw_csv_path, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date() # Ensure date object for comparison
    
    bronze_feature_directory = os.path.join(bronze_base_dir, feature_name)
    if not os.path.exists(bronze_feature_directory):
        os.makedirs(bronze_feature_directory)

    # load data - IRL ingest from back end source system
    # Read the entire CSV then filter by snapshot_date
    df = spark.read.csv(raw_csv_path, header=True, inferSchema=True)
    
    # Filter by snapshot_date. Ensure correct date comparison.
    # Cast snapshot_date in DataFrame to DateType if it's string after inferSchema
    if 'snapshot_date' in df.columns and str(df.schema['snapshot_date'].dataType) != 'DateType()':
        df = df.withColumn("snapshot_date", col("snapshot_date").cast("date"))
        
    df_filtered = df.filter(col('snapshot_date') == snapshot_date)
    
    print(f"{feature_name} - {snapshot_date_str} raw row count: {df.count()}, filtered row count: {df_filtered.count()}")

    # save bronze table to datamart
    partition_name = f"bronze_{feature_name}_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(bronze_feature_directory, partition_name)
    
    # Using toPandas().to_csv() as per existing pattern, though not optimal for large data
    df_filtered.toPandas().to_csv(filepath, index=False)
    print(f'Saved {feature_name} bronze to: {filepath}')
    return df_filtered

def process_bronze_attributes(snapshot_date_str, bronze_features_directory, spark):
    raw_csv_path = "data/features_attributes.csv"
    return process_feature_to_bronze(snapshot_date_str, bronze_features_directory, "attributes", raw_csv_path, spark)

def process_bronze_financials(snapshot_date_str, bronze_features_directory, spark):
    raw_csv_path = "data/features_financials.csv"
    return process_feature_to_bronze(snapshot_date_str, bronze_features_directory, "financials", raw_csv_path, spark)

def process_bronze_clickstream(snapshot_date_str, bronze_features_directory, spark):
    raw_csv_path = "data/feature_clickstream.csv"
    return process_feature_to_bronze(snapshot_date_str, bronze_features_directory, "clickstream", raw_csv_path, spark)

