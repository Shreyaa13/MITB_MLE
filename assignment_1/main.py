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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_features
import utils.data_processing_silver_features
import utils.data_processing_gold_features   


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

# --- FEATURE STORE PIPELINE ---
print("\n--- STARTING FEATURE STORE PIPELINE ---")

# Create bronze datalake for Features
bronze_features_directory = "datamart/bronze/features/"
# Subdirectories for attributes, financials, clickstream will be created by functions if not exist

# # Run bronze backfill for Features
for date_str in dates_str_lst:
    print(f"\nProcessing Features Bronze for {date_str}")
    utils.data_processing_bronze_features.process_bronze_attributes(date_str, bronze_features_directory, spark)
    utils.data_processing_bronze_features.process_bronze_financials(date_str, bronze_features_directory, spark)
    utils.data_processing_bronze_features.process_bronze_clickstream(date_str, bronze_features_directory, spark)

# Create silver datalake for Features
silver_features_directory = "datamart/silver/features/"
# Subdirectories for attributes, financials, clickstream will be created by functions if not exist

# Run silver backfill for Features
for date_str in dates_str_lst:
    print(f"\nProcessing Features Silver for {date_str}")
    utils.data_processing_silver_features.process_silver_attributes(date_str, bronze_features_directory, silver_features_directory, spark)
    utils.data_processing_silver_features.process_silver_financials(date_str, bronze_features_directory, silver_features_directory, spark)
    utils.data_processing_silver_features.process_silver_clickstream(date_str, bronze_features_directory, silver_features_directory, spark)

# Create gold datalake for Feature Store
gold_feature_store_directory = "datamart/gold/feature_store/"
if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)

# Run gold backfill for Feature Store
for date_str in dates_str_lst:
    print(f"\nProcessing Feature Store Gold for {date_str}")
    utils.data_processing_gold_features.process_feature_store_gold_table(date_str, silver_features_directory, gold_feature_store_directory, spark)

print("\n--- FEATURE STORE PIPELINE COMPLETED ---")


print("\n--- GOLD FEATURE STORE SAMPLE ---")
feature_files_list = [os.path.join(gold_feature_store_directory, os.path.basename(f)) for f in glob.glob(os.path.join(gold_feature_store_directory, '*.parquet'))]
if feature_files_list:
    df_features_gold = spark.read.option("header", "true").parquet(*feature_files_list)
    print("Feature Store Gold")
    df_features_gold.printSchema()
    df_features_gold.show(5)

spark.stop()
print("\n--- SCRIPT COMPLETED ---")


    