import argparse
import os
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
from pyspark.sql import Window

from pyspark.sql.functions import col, last, first, coalesce
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import sys


# Adapted from model_inference.ipynb notebook - refer to that for observations"

def main(snapshotdate, modelname):
    try:
        print('\n\n---starting job---\n\n')
        
        # Initialize SparkSession
        spark = pyspark.sql.SparkSession.builder \
            .appName("dev") \
            .master("local[*]") \
            .getOrCreate()
        
        # Set log level to ERROR to hide warnings
        spark.sparkContext.setLogLevel("ERROR")
        
        # --- set up config ---
        config = {}
        config["snapshot_date_str"] = snapshotdate
        config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
        config["model_name"] = modelname
        config["model_bank_directory"] = "./model_bank/"
        config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"]
        
        pprint.pprint(config)
        
        # Load the model from the pickle file
        with open(config["model_artefact_filepath"], 'rb') as file:
            model_artefact = pickle.load(file)
        
        print("Model loaded successfully! " + config["model_artefact_filepath"])

        # --- load feature store ---
        cust_fin_risk_dir = "./datamart/gold/feature_store/cust_fin_risk/"

        partition_name = "gold_ft_store_cust_fin_risk_" + config["snapshot_date_str"].replace('-','_') + '.parquet'
        filepath = cust_fin_risk_dir + partition_name
        cust_fin_risk_df = spark.read.parquet(filepath)
        print('loaded from:', filepath, 'row count:', cust_fin_risk_df.count())


        eng_dir = "./datamart/gold/feature_store/eng/"

        partition_name = "gold_ft_store_engagement_" + config["snapshot_date_str"].replace('-','_') + '.parquet'
        filepath = eng_dir + partition_name
        eng_df = spark.read.parquet(filepath)
        print('loaded from:', filepath, 'row count:', eng_df.count())

        features_df = cust_fin_risk_df.join(eng_df, on=["Customer_ID", "snapshot_date"], how="left")

        # Define columns to fill
        feature_cols = [
            "click_1m", "click_2m", "click_3m", "click_4m", "click_5m", "click_6m",
            "Credit_History_Age", "Num_Fin_Pdts", "EMI_to_Salary", "Debt_to_Salary",
            "Repayment_Ability", "Loans_per_Credit_Item", "Loan_Extent", "Outstanding_Debt",
            "Interest_Rate", "Delay_from_due_date", "Changed_Credit_Limit"
        ]

        # Window for forward fill
        fwd_window = Window.partitionBy("Customer_ID").orderBy("snapshot_date").rowsBetween(Window.unboundedPreceding, 0)

        # Window for backward fill
        bwd_window = Window.partitionBy("Customer_ID").orderBy("snapshot_date").rowsBetween(0, Window.unboundedFollowing)

        # Apply fills
        for col_name in feature_cols:
            fwd_fill = last(col(col_name), ignorenulls=True).over(fwd_window)
            bwd_fill = first(col(col_name), ignorenulls=True).over(bwd_window)
            features_df = features_df.withColumn(col_name, coalesce(fwd_fill, bwd_fill))

        features_df_2 = features_df.toPandas()

        print(features_df_2)

        X_inference = features_df_2[feature_cols]

        transformer_stdscaler = model_artefact["preprocessing_transformers"]["stdscaler"]
        X_inference = transformer_stdscaler.transform(X_inference)

        # load model
        model = model_artefact["model"]
        
        # predict model
        y_inference = model.predict_proba(X_inference)[:, 1]
        
        # prepare output
        y_inference_pdf = features_df_2[["Customer_ID","snapshot_date"]].copy()
        y_inference_pdf["model_name"] = config["model_name"]
        y_inference_pdf["model_predictions"] = y_inference
        

        # --- save model inference to datamart gold table ---
        # create gold datalake
        gold_dir = f"datamart/gold/model_predictions/{config['model_name'][:-4]}/"
        print(gold_dir)
        
        if not os.path.exists(gold_dir):
            os.makedirs(gold_dir)
        
        # save gold table 
        partition_name = config["model_name"][:-4] + "_predictions_" + config["snapshot_date_str"].replace('-','_') + '.parquet'
        filepath = gold_dir + partition_name
        spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
        print('saved to:', filepath)
        
        # --- end spark session --- 
        spark.stop()
        
        print('\n\n---completed job---\n\n')

    except Exception as e:
        print(e)


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model_name")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.modelname)
