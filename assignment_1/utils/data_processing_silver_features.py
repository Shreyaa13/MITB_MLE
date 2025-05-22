import os
import re
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DoubleType

def _parse_credit_history_age_func(age_str):
    if age_str is None:
        return None
    try:
        years = 0
        months = 0
        year_match = re.search(r"(\d+)\s*Years?", str(age_str))
        month_match = re.search(r"(\d+)\s*Months?", str(age_str))
        if year_match:
            years = int(year_match.group(1))
        if month_match:
            months = int(month_match.group(1))
        return years * 12 + months
    except Exception:
        return None

parse_credit_history_age_udf = F.udf(_parse_credit_history_age_func, IntegerType())

def process_silver_attributes(snapshot_date_str, bronze_features_dir, silver_features_dir, spark):
    bronze_file_path = os.path.join(bronze_features_dir, "attributes", f"bronze_attributes_{snapshot_date_str.replace('-', '_')}.csv")
    silver_attributes_directory = os.path.join(silver_features_dir, "attributes")
    if not os.path.exists(silver_attributes_directory):
        os.makedirs(silver_attributes_directory)

    df = spark.read.csv(bronze_file_path, header=True, inferSchema=True)
    print(f'Loaded attributes from: {bronze_file_path}, row count: {df.count()}')

    # Clean data: enforce schema / data type
    df = df.withColumn("Customer_ID", F.col("Customer_ID").cast(StringType())) \
           .withColumn("Occupation", F.col("Occupation").cast(StringType())) \
           .withColumn("snapshot_date", F.to_date(F.col("snapshot_date"), "yyyy-MM-dd")) # Ensure date type

    # Clean Age: remove underscores, cast to int
    df = df.withColumn("Age", F.regexp_replace("Age", "_", "").cast(IntegerType()))

    # Logging: invalid ages (outside 18–100)
    invalid_age_range_count = df.filter((F.col("Age") < 18) | (F.col("Age") > 100)).count()
    print(f"Ages outside valid range (18–100): {invalid_age_range_count}")

    # Impute invalid ages using median of valid values
    median_age = df.filter((F.col("Age").isNotNull()) &
                           (F.col("Age") >= 18) &
                           (F.col("Age") <= 100)) \
                   .approxQuantile("Age", [0.5], 0.01)[0]
    
    df = df.withColumn("Age",
        F.when((F.col("Age").isNull()) | (F.col("Age") < 18) | (F.col("Age") > 100),
               F.lit(int(median_age)))
         .otherwise(F.col("Age"))
    )

    # Logging: invalid SSNs (contains non-digit characters other than '-')
    ssn_invalid_count = df.filter(~F.col("SSN").rlike(r"^\d{3}-\d{2}-\d{4}$")).count()
    print(f"Invalid SSNs (special/random characters): {ssn_invalid_count}")
   
    # Clean Occupation: replace only if it's a string of multiple underscores (e.g., "___")
    df = df.withColumn("Occupation",
        F.when(F.col("Occupation").rlike(r"^_{3,}$"), "Unknown")  # matches 2 or more underscores only
         .otherwise(F.col("Occupation"))
    )

    # Log: placeholder Occupations (underscores only)
    occupation_placeholder_count = df.filter(F.col("Occupation") == "Unknown").count()
    print(f"Occupations replaced with 'Unknown': {occupation_placeholder_count}")

    df = df.select("Customer_ID", "Age", "Occupation", "snapshot_date") # Final columns selected
    
    # save silver table
    partition_name = f"silver_attributes_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_attributes_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print(f'Saved silver attributes to: {filepath}, row count: {df.count()}')
    return df

def process_silver_financials(snapshot_date_str, bronze_features_dir, silver_features_dir, spark):
    bronze_file_path = os.path.join(bronze_features_dir, "financials", f"bronze_financials_{snapshot_date_str.replace('-', '_')}.csv")
    silver_financials_directory = os.path.join(silver_features_dir, "financials")
    if not os.path.exists(silver_financials_directory):
        os.makedirs(silver_financials_directory)

    df = spark.read.csv(bronze_file_path, header=True, inferSchema=True) # Keep inferSchema for now, cast explicitly
    print(f'Loaded financials from: {bronze_file_path}, row count: {df.count()}')

    # Clean and cast columns
    df = df.withColumn("Customer_ID", F.col("Customer_ID").cast(StringType())) \
           .withColumn("Annual_Income", F.regexp_replace(F.col("Annual_Income"), "[^0-9.]", "").cast(FloatType())) \
           .withColumn("Monthly_Inhand_Salary", F.col("Monthly_Inhand_Salary").cast(FloatType())) \
           .withColumn("Num_Bank_Accounts", F.col("Num_Bank_Accounts").cast(IntegerType())) \
           .withColumn("Num_Credit_Card", F.col("Num_Credit_Card").cast(IntegerType())) \
           .withColumn("Interest_Rate", F.col("Interest_Rate").cast(IntegerType())) \
           .withColumn("Num_of_Loan_clean", F.regexp_replace(F.col("Num_of_Loan"), "[^0-9]", "").cast(IntegerType())) \
           .withColumn("Derived_Num_of_Loan", F.when(F.col("Type_of_Loan").isNotNull(), F.size(F.split(F.col("Type_of_Loan"), ",")))
                       .otherwise(0)) \
           .withColumn("Num_of_Loan", F.when((F.col("Num_of_Loan_clean").isNull()) | (F.col("Num_of_Loan_clean") < 0) 
                                             | (F.col("Num_of_Loan_clean") > 50), 
                       F.col("Derived_Num_of_Loan")).otherwise(F.col("Num_of_Loan_clean"))
            ).drop("Derived_Num_of_Loan").drop("Num_of_Loan_clean") \
           .withColumn("Type_of_Loan", F.when(F.col("Num_of_Loan") == 0, "No Loan")
                 .otherwise(F.when((F.col("Type_of_Loan") == "") | F.col("Type_of_Loan").isNull(), None)
                 .otherwise(F.col("Type_of_Loan")))) \
           .withColumn("Delay_from_due_date", F.col("Delay_from_due_date").cast(IntegerType())) \
           .withColumn("Num_of_Delayed_Payment", F.regexp_replace(F.col("Num_of_Delayed_Payment"), "[^0-9]", "").cast(IntegerType())) \
           .withColumn("Num_of_Delayed_Payment", F.when(F.col("Num_of_Delayed_Payment") < 0, None) 
                       .otherwise(F.col("Num_of_Delayed_Payment").cast(IntegerType()))) \
           .withColumn("Changed_Credit_Limit_Clean", F.when((F.col("Changed_Credit_Limit") == "_") | 
                (F.col("Changed_Credit_Limit").cast(FloatType()) < 0), None).otherwise(F.col("Changed_Credit_Limit").cast(FloatType()))) \
           .withColumn("Num_Credit_Inquiries", F.col("Num_Credit_Inquiries").cast(IntegerType())) \
           .withColumn("Credit_Mix", F.when(F.col("Credit_Mix") == "_", "Unknown").otherwise(F.col("Credit_Mix")).cast(StringType())) \
           .withColumn("Outstanding_Debt", F.regexp_replace(F.col("Outstanding_Debt"), "[^0-9.]", "").cast(FloatType())) \
           .withColumn("Credit_Utilization_Ratio", F.col("Credit_Utilization_Ratio").cast(FloatType())) \
           .withColumn("Credit_History_Age_Months", parse_credit_history_age_udf(F.col("Credit_History_Age"))) \
           .withColumn("Payment_of_Min_Amount", F.col("Payment_of_Min_Amount").cast(StringType())) \
           .withColumn("Total_EMI_per_month", F.col("Total_EMI_per_month").cast(FloatType())) \
           .withColumn("Amount_invested_monthly", F.regexp_replace(F.col("Amount_invested_monthly"), "[^0-9.]", "").cast(FloatType())) \
           .withColumn("Payment_Behaviour", F.when(F.col("Payment_Behaviour").rlike("^[A-Za-z_]+$"), F.col("Payment_Behaviour"))
                 .otherwise("Unknown")) \
           .withColumn("Monthly_Balance", F.regexp_replace(F.col("Monthly_Balance"), "[^0-9.]", "").cast(FloatType())) \
           .withColumn("snapshot_date", F.to_date(F.col("snapshot_date"), "yyyy-MM-dd")) \
           .withColumn("EMI_to_Monthly_Income_Ratio", F.round(F.col("Total_EMI_per_month") / F.col("Monthly_Inhand_Salary"),3)) \
           .withColumn("Debt_to_Monthly_Income_Ratio", F.round(F.col("Outstanding_Debt") / F.col("Monthly_Inhand_Salary"),3))

    median_changed_credit_limit = df.approxQuantile("Changed_Credit_Limit_Clean", [0.5], 0.01)[0]
    df = df.withColumn("Changed_Credit_Limit", F.round(F.when(F.col("Changed_Credit_Limit_Clean").isNull(), F.lit(median_changed_credit_limit))
         .otherwise(F.col("Changed_Credit_Limit_Clean")),2)).drop("Changed_Credit_Limit_Clean") \

    # Select relevant columns, drop original complex string columns if parsed
    df = df.select(
        "Customer_ID", "snapshot_date", "Annual_Income", "Monthly_Inhand_Salary",
        "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Type_of_Loan",
        "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
        "Num_Credit_Inquiries", "Credit_Mix", "Outstanding_Debt", "Credit_Utilization_Ratio",
        "Credit_History_Age", "Credit_History_Age_Months", "Payment_of_Min_Amount", "Total_EMI_per_month",
        "Amount_invested_monthly", "Payment_Behaviour", "Monthly_Balance", "EMI_to_Monthly_Income_Ratio",
        "Debt_to_Monthly_Income_Ratio"
    )
    
    # save silver table
    partition_name = f"silver_financials_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_financials_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print(f'Saved silver financials to: {filepath}')
    return df

def process_silver_clickstream(snapshot_date_str, bronze_features_dir, silver_features_dir, spark):
    bronze_file_path = os.path.join(bronze_features_dir, "clickstream", f"bronze_clickstream_{snapshot_date_str.replace('-', '_')}.csv")
    silver_clickstream_directory = os.path.join(silver_features_dir, "clickstream")
    if not os.path.exists(silver_clickstream_directory):
        os.makedirs(silver_clickstream_directory)

    df = spark.read.csv(bronze_file_path, header=True, inferSchema=True)
    print(f'Loaded clickstream from: {bronze_file_path}, row count: {df.count()}')

    # Ensure snapshot_date is DateType
    df = df.withColumn("snapshot_date", F.to_date(F.col("snapshot_date"), "yyyy-MM-dd"))

    # Cast all fe_X columns to IntegerType just to be sure, though inferSchema might get them right
    for i in range(1, 21):
        col_name = f"fe_{i}"
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast(IntegerType()))
    
    df = df.withColumn("Customer_ID", F.col("Customer_ID").cast(StringType()))

    # save silver table
    partition_name = f"silver_clickstream_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_clickstream_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print(f'Saved silver clickstream to: {filepath}, row count: {df.count()}')
    return df