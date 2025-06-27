import logging
from typing import List
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql import DataFrame
from pyspark.sql import types
from pyspark.sql.types import FloatType
from pyspark.sql.functions import year, col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, RegexTokenizer
from pyspark.sql.functions import lower, regexp_replace
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
from google.cloud import storage
import os

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def spark_init(key_file="../keys/arxiv-trends-key.json"):

    # to upload to bigquery
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName('pyspark-bigquery-upload') \
        .config("spark.jars", "/opt/spark/jars/gcs-connector-hadoop3-latest.jar,/opt/spark/jars/spark-bigquery-with-dependencies.jar") \
        .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
        .config("spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS") \
        .config("spark.hadoop.google.cloud.auth.service.account.enable", "true") \
        .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", 
                os.getenv('GOOGLE_APPLICATION_CREDENTIALS')) \
        .config("spark.sql.repl.eagerEval.enabled", True) \
        .getOrCreate()

    
    # Authentication 
    spark.conf.set("google.cloud.auth.service.account.json.keyfile", key_file)
    spark.conf.set("parentProject", "arxiv-trends")

    # need this line to avoid columns with type arrays not being present when copying to bigquery
    spark.conf.set("spark.datasource.bigquery.intermediateFormat", "orc")

    # Set GCS credentials if necessary
    spark._jsc.hadoopConfiguration().set("google.cloud.auth.service.account.json.keyfile", key_file)

    return spark



def set_spark_schema(spark, file_name='gs://arxiv-bucket-111865037319658112231/arxiv_data/arxiv_hep-ex_papers_2012_2018.parquet'):
    # set the schema for the columns
    schema = types.StructType([
        types.StructField('title', types.StringType(), True),
        types.StructField('summary', types.StringType(), True),
        types.StructField('submission_date', types.DateType(), True),
        types.StructField('id', types.StringType(), True),
        types.StructField('author', types.ArrayType(types.StringType()), True),
        types.StructField('primary_category', types.StringType(), True),
        types.StructField('categories',  types.ArrayType(types.StringType()), True)
    ])

    # set the correct data types
    df = spark.read \
        .schema(schema) \
        .parquet(file_name)
    return df
    
# Define UDF to clean text
def clean_text(text: str) -> List[str]:
    """
    Clean and preprocess text by tokenizing, removing stopwords, and lemmatizing.
    
    Args:
        text: Raw text string to clean
        
    Returns:
        List of cleaned and lemmatized words
    """
    if not isinstance(text, str):
        logger.warning(f"Expected string but got {type(text)}. Converting to string.")
        text = str(text)
        
    text = text.lower()  # Convert to lowercase
    
    # Tokenize text into words
    try:
        words = word_tokenize(text)
    except Exception as e:
        logger.error(f"Error tokenizing text: {e}")
        return []
    
    # Remove stopwords and lemmatize
    cleaned_words = [
        lemmatizer.lemmatize(word) for word in words 
        if word.isalpha() and word not in stop_words
    ]
    
    return cleaned_words



def process_column_spark_native(df: DataFrame, column: str) -> DataFrame:
    """
    Clean the specified column using Spark's native text processing functions.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    cleaned_column = f'cleaned_{column}'
    
    # Convert to lowercase and remove non-alphabetic characters
    df = df.withColumn(f"{column}_lower", lower(col(column)))
    df = df.withColumn(f"{column}_clean", regexp_replace(col(f"{column}_lower"), "[^a-zA-Z\\s]", ""))
    
    # Tokenize
    tokenizer = RegexTokenizer(
        inputCol=f"{column}_clean", 
        outputCol=f"{column}_tokens",
        pattern="\\s+",  # Split on whitespace
        minTokenLength=2  # Remove very short tokens
    )
    df = tokenizer.transform(df)
    
    # Remove stopwords
    stopwords_remover = StopWordsRemover(
        inputCol=f"{column}_tokens", 
        outputCol=cleaned_column
    )
    df = stopwords_remover.transform(df)
    
    # Clean up intermediate columns
    df = df.drop(f"{column}_lower", f"{column}_clean", f"{column}_tokens")
    
    return df


def process_column(df: DataFrame, column: str) -> DataFrame:    
    """
    Clean the specified column and create a new column with the processed text.
    
    Args:
        df: DataFrame containing arXiv papers
        column: The column to clean ('title' or 'summary')
        
    Returns:
        DataFrame with added cleaned text column
        
    Raises:
        ValueError: If the specified column is not in the DataFrame
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    cleaned_column = f'cleaned_{column}'
    
    # Define a UDF to apply your clean_text function
    clean_text_udf = udf(clean_text, ArrayType(StringType()))
    
    logger.info(f"Processing column: {column}")
    df = df.withColumn(cleaned_column, clean_text_udf(col(column)))
    return df



def calculate_sentiment(df: DataFrame, column_name: str) -> DataFrame:
    """
    Calculate sentiment polarity score for text in the specified column in a Spark DataFrame.

    Args:
        df: Spark DataFrame containing arXiv papers
        column_name: Column name to analyze sentiment from

    Returns:
        Spark DataFrame with added 'sentiment' column

    Raises:
        ValueError: If the specified column is not in the DataFrame
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    
    sentiment_column = f'sentiment_{column_name}'

    def compute_sentiment(text):
        if text is None:
            return 0.0
        try:
            return TextBlob(str(text)).sentiment.polarity
        except Exception:
            return 0.0

    sentiment_udf = udf(compute_sentiment, FloatType())
    
    logger.info(f"Sentiment analysis completed for column: {column_name}")
    df = df.withColumn(sentiment_column, sentiment_udf(col(column_name)))
    return df


def check_gcs_file_exists(bucket_name, gcs_file_name):
    client = storage.Client(project="arxiv-trends")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_file_name)
    return blob.exists()

def save_file_gcs(df: DataFrame, output_gcs_path="gs://test-arxiv-bucket-111865037319658112231/arxiv_data/processed/arxiv_hep-ex_papers_2012_2018_cleaned.parquet"):
    """
    Saves the Spark DataFrame back in GCS.
    
    Parameters:
    - df: Spark DataFrame to save
    - output_gcs_path: GCS path to save the Parquet file
    """
    # since files are small, don't partition
    df = df.coalesce(1)

    # df.show()
    df.write.mode("overwrite").parquet(output_gcs_path)
    print(f"Data saved to: {output_gcs_path}")


def save_file_bigquery(df: DataFrame, output_table="arxiv-trends.arxiv_papers.test_table"):
    """
    Saves the Spark DataFrame in BigQuery
    
    Parameters:
    - df: Spark DataFrame to save
    - output_table: name of the table in BigQuery
    """
    # since files are small, don't partition
    df = df.coalesce(1)

    # df.show()
    df.write \
    .format("bigquery") \
    .option("table", output_table) \
    .option("temporaryGcsBucket", "temp-bucket-111865037319658112231") \
    .mode("overwrite") \
    .save()


def spark_stop(spark):
    print('Stopping spark session')
    