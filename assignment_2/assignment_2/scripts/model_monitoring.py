import argparse
import os
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import sys
print(sys.version)

def psi_calculation(expected, actual, buckets=10):
    # Convert to numeric and drop non-convertible values
    expected = pd.to_numeric(expected, errors='coerce')
    actual = pd.to_numeric(actual, errors='coerce')

    # Drop NaNs after conversion
    expected = expected.dropna()
    actual = actual.dropna()

    # Create breakpoints from expected
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) <= 2:
        raise ValueError("Unique values count not enough to form bins.")

    # Bin data
    expected_bins = pd.cut(expected, bins=breakpoints, include_lowest=True)
    actual_bins = pd.cut(actual, bins=breakpoints, include_lowest=True)

    # Get distributions
    expected_dist = pd.value_counts(expected_bins, normalize=True).sort_index()
    actual_dist = pd.value_counts(actual_bins, normalize=True).sort_index()

    # Align bins and avoid division by zero
    expected_dist, actual_dist = expected_dist.align(actual_dist, fill_value=1e-6)

    # PSI calculation
    psi = ((actual_dist - expected_dist) * np.log(actual_dist / expected_dist)).sum()

    return psi

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

        print('\Checking prediction drift using PSI...')

        psi_value = None
    
        # --- load current prediction store ---
        predictions_dir = f"./datamart/gold/model_predictions/{modelname[:-4]}/"

        current_partition_name = f'{modelname[:-4]}' + '_predictions_' + f'{snapshotdate}'.replace('-','_') + '.parquet'
        filepath = predictions_dir + current_partition_name
        predictions_df = spark.read.parquet(filepath)
        print('loaded from:', filepath, 'row count:', predictions_df.count())

        predictions_df = predictions_df.toPandas()
        new_predict_arr = predictions_df["model_predictions"]

        # --- load last month's prediction ---
        date_obj = pd.to_datetime(snapshotdate)
        previous_predict_date = (date_obj - pd.DateOffset(months=1)).strftime("%Y-%m-%d")

        prev_partition_name = f'{modelname[:-4]}' + '_predictions_' + f'{previous_predict_date}'.replace('-','_') + '.parquet'
        filepath = predictions_dir + prev_partition_name
        previous_predictions_df = spark.read.parquet(filepath)
        print('loaded from:', filepath, 'row count:', predictions_df.count())

        previous_predictions_df = previous_predictions_df.toPandas()
        previous_predict_arr = previous_predictions_df["model_predictions"]
        
        # --- Calculate psi ---
        psi_value = psi_calculation(previous_predict_arr, new_predict_arr)
        print(psi_value)

        # PSI interpretation
        if psi_value > 0.25:
            print(f'[ALERT] PSI = {psi_value:.4f} CRITICAL: Significant population shift detected!')
        elif psi_value > 0.1:
            print(f'[WARNING] PSI = {psi_value:.4f} WARNING: Moderate population shift detected')
        else:
            print(f'[OK] PSI = {psi_value:.4f} STABLE: Predictions are stable')

        monitor_directory = f"./datamart/gold/monitor/model_monitor/"

        if not os.path.exists(monitor_directory):
            os.makedirs(monitor_directory)

        kde_plot_dir = os.path.join(monitor_directory, "KDE/")
        os.makedirs(kde_plot_dir, exist_ok=True)

        # --- Plot distribution shift using KDE ---
        print('Generating KDE plot of prediction distributions...')

        # Plot KDE of predictions of previous vs current
        sns.kdeplot(previous_predict_arr, label= previous_predict_date, color = 'blue', shade=True)
        sns.kdeplot(new_predict_arr, label= snapshotdate, color = 'green', shade=True)

        # Plot
        plt.legend()
        plt.title(f'Prediction Distribution KDE: {previous_predict_date} vs {snapshotdate}'.replace('-','_'))
        plt.xlabel("Model Prediction Value")
        plt.ylabel("Density")

        partition_name = f'{modelname[:-4]}' + '_KDE_predictions_' + f'{snapshotdate}'.replace('-','_') + '.png'
        filepath = kde_plot_dir + partition_name

        # Save the figure
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close('all')

        # ----- Monitoring Model Performance Metrics -----

        label_store_dir = f"./datamart/gold/label_store/"
        psi_plot_dir = os.path.join(monitor_directory, "psi/")
        auc_plot_dir = os.path.join(monitor_directory, "roc_auc/")
        snapshot_csv_dir = os.path.join(monitor_directory, "snapshots/")

        os.makedirs(psi_plot_dir, exist_ok=True)
        os.makedirs(auc_plot_dir, exist_ok=True)
        os.makedirs(snapshot_csv_dir, exist_ok=True)

        partition_name = 'gold_label_store_' +  f'{snapshotdate}'.replace('-','_') + '.parquet'
        filepath = label_store_dir + partition_name
        label_df = spark.read.parquet(filepath)
        print('loaded labels from:', filepath, 'row count:', label_df.count())

        label_df = label_df.toPandas()
        predictions_only = predictions_df[["model_predictions"]].copy()

         # Merge predictions with labels
        combined_df = label_df.copy()
        combined_df['model_predictions'] = predictions_only['model_predictions']
        combined_df = combined_df[['label', 'model_predictions']].dropna()

        print(combined_df.head(20))

        roc_auc = None

        if len(combined_df['label']) > 0:
            roc_auc = roc_auc_score(combined_df['label'], combined_df["model_predictions"])
            if roc_auc< 0.65:
                print(f'roc_auc: {roc_auc}, ALERT : Model needs retraining')
            else:
                print(f'roc_auc: {roc_auc}, CLEAR : Model performance ok')
        else:
            print('labels not available!')

        # Save metrics snapshot
        snapshot_df = pd.DataFrame({
            'snapshotdate': [snapshotdate],
            'modelname': [modelname],
            'predict_psi': [psi_value],
            'roc_auc': [roc_auc]
        })

        partition_name = "monitor_" + f'{modelname[:-4]}_'+ snapshotdate.replace('-','_') + '.csv'
        filepath = snapshot_csv_dir + partition_name
        snapshot_df.to_csv(filepath)

        # Load historical snapshots
        all_snapshot_csvs = glob.glob(os.path.join(snapshot_csv_dir, '*.csv'))
        history_df = pd.concat([pd.read_csv(f) for f in all_snapshot_csvs], ignore_index=True)
        history_df['snapshotdate'] = pd.to_datetime(history_df['snapshotdate'])
        history_df = history_df.sort_values(by='snapshotdate').reset_index(drop=True)

        # Plot PSI trend
        if len(history_df['predict_psi']) > 0:
            plt.figure()
            plt.plot(history_df['snapshotdate'], history_df['predict_psi'], marker='o')
            plt.title(f'PSI Trend: {snapshotdate}')
            plt.xlabel('Snapshot Date')
            plt.ylabel('Prediction PSI')
            plt.xticks(rotation=90)

            psi_plot_path = os.path.join(psi_plot_dir, f'psi_trend_{modelname[:-4]}_{snapshotdate.replace("-", "_")}.png')
            plt.tight_layout()
            plt.savefig(psi_plot_path, dpi=300, bbox_inches='tight')
            plt.close('all')

        # Plot ROC AUC trend
        if len(history_df['roc_auc']) > 0:
            plt.figure()
            plt.plot(history_df['snapshotdate'], history_df['roc_auc'], marker='o', color='green')
            plt.title(f'ROC AUC Trend: {snapshotdate}')
            plt.xlabel('Snapshot Date')
            plt.ylabel('ROC AUC')
            plt.xticks(rotation=90)

            auc_plot_path = os.path.join(auc_plot_dir, f'roc_auc_trend_{modelname[:-4]}_{snapshotdate.replace("-", "_")}.png')
            plt.tight_layout()
            plt.savefig(auc_plot_path, dpi=300, bbox_inches='tight')
            plt.close('all')
        
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
