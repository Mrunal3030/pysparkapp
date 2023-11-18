from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create a Spark session
spark = SparkSession.builder.appName("DataAnalysisSparkApp").getOrCreate()

# Read CSV file into a Spark DataFrame
csv_file_path = "D:/pyspark application/zipcodes.csv"  # Replace with your actual file path
df_csv = spark.read.csv(csv_file_path, header=True, inferSchema=True)

# Read Parquet file into a Spark DataFrame
parquet_file_path = "D:/pyspark application/userdata1.parquet"  # Replace with your actual file path
df_parquet = spark.read.parquet(parquet_file_path)

# Data Exploration
# Display basic statistics for CSV file
df_csv.describe().show()

# Display basic statistics for Parquet file
df_parquet.describe().show()

# Visualization
# Example: Box plot using Seaborn
sns.boxplot(x=df_csv.toPandas()['LocationText'])
plt.title('Box Plot for CSV Data')
plt.show()

# Metrics Comparison
# Assuming you have a common column in both CSV and Parquet files for comparison
common_column = "LocationText"

# # Extract metrics for CSV and Parquet files
# csv_metric = df_csv.groupBy(common_column).agg({"metric_column": "mean"}).collect()
# exprs = [sum(common_column)]

# Using agg to perform the aggregation
# jdf = df_csv.agg(*exprs)

# Show the result
# jdf.show()

# # parquet_metric = df_parquet.groupBy(common_column).agg({"metric_column": "mean"}).collect()

# # Convert Spark results to Pandas for easier comparison
# df_csv_metric = pd.DataFrame(csv_metric, columns=[common_column, 'csv_mean_metric'])
# df_parquet_metric = pd.DataFrame(parquet_metric, columns=[common_column, 'parquet_mean_metric'])

# # Display metrics comparison
# comparison_df = pd.merge(df_csv_metric, df_parquet_metric, on=common_column)
# print(comparison_df)

# Stop the Spark session
spark.stop()
