from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from functools import reduce
import os
import sys
import logging
import tempfile

import nltk
nltk.download('stopwords')

# Add the modules folder to Python path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../modules')))


# Import your local modules
from fetching import fetch_papers_by_years
from upload_file_gcs import upload_file_to_gcs
from processing_spark import (
    spark_init,
    check_gcs_file_exists,
    set_spark_schema,
    process_column,
    calculate_sentiment,
    save_file_bigquery,
    process_column_spark_native
)

# DAG parameters
DOMAINS = ['cs.SY']
START_YEAR = 2000
END_YEAR = 2025

BUCKET_NAME = os.getenv('BUCKET_NAME')
JAVA_HOME = os.getenv('JAVA_HOME')
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

# Use Airflow's temporary directory
TMP_DIR = tempfile.gettempdir()

# Default Airflow arguments
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG Definition
dag = DAG(
    'arxiv_etl_pipeline',
    default_args=default_args,
    description='Fetch, process and load arXiv papers to GCS and BigQuery',
    schedule=None,
    catchup=False,
    tags=['arxiv', 'etl', 'gcs', 'bigquery']
)

# Task 1: Fetch papers
def fetch_data():
    for domain in DOMAINS:
        for year in range(START_YEAR, END_YEAR):
            fetch_papers_by_years(
                start_year=year,
                end_year=year+1,
                domain=domain,
                file_path=TMP_DIR
            )

# Task 2: Upload to GCS
def upload_data():
    for domain in DOMAINS:
        for year in range(START_YEAR, END_YEAR):
            local_file = f'{TMP_DIR}/arxiv_{domain}_papers_{year}-{year+1}.parquet'
            blob_name = f'arxiv_data/{domain}/arxiv_{domain}_papers_{year}-{year+1}.parquet'
            if os.path.exists(local_file):
                upload_file_to_gcs(BUCKET_NAME, local_file, blob_name)




# Task 3: Process in Spark and load to BigQuery
def transform_and_load():
    spark = spark_init(key_file=GOOGLE_APPLICATION_CREDENTIALS)
    print('Done initializing pyspark')
    for domain in DOMAINS:
        dataframes = []
        for year in range(START_YEAR, END_YEAR):
            gcs_file_name = f'arxiv_data/{domain}/arxiv_{domain}_papers_{year}-{year+1}.parquet'
            print('Checking if the file already exists in GCP')
        if check_gcs_file_exists(bucket_name=BUCKET_NAME, gcs_file_name=gcs_file_name):
            # pyspark processing
            print('Setting spark schema')
            df = set_spark_schema(spark, file_name=f'gs://{BUCKET_NAME}/{gcs_file_name}')
            print('Processing columns')
            #df = process_column(df, 'title')
            #df = process_column(df, 'summary')
            df = process_column_spark_native(df, 'title')
            df = process_column_spark_native(df, 'summary')
            df = calculate_sentiment(df, 'title')
            df = calculate_sentiment(df, 'summary')
            dataframes.append(df)
        else:
            print(f"File gs://{BUCKET_NAME}/{gcs_file_name} does not exist, check if it should.")
            continue
            # merge the dataframes
    print('Merging dataframes')
    if dataframes:
        merged_df = reduce(lambda df1, df2: df1.unionByName(df2), dataframes)
    else:
        merged_df = None
    
    print('Proceeding to save files in BigQuery')
    # save the file
    if merged_df:
        # BigQuery does not accept table names with dashes, have to replace them
        DOMAIN_CLEANED = domain.replace("-", "_")
        DOMAIN_CLEANED = DOMAIN_CLEANED.replace(".", "_")
        # attaching the domain at the end so later we can use wildcards in querying in BigQuery
        #output_table = f'arxiv-trends.arxiv_papers.arxiv_{DOMAIN_CLEANED}_papers_{START_YEAR}_{END_YEAR}'
        output_table = f'arxiv-trends.arxiv_papers.arxiv_papers_{START_YEAR}_{END_YEAR}_{DOMAIN_CLEANED}'
        #print(df.printSchema())
        #print(df.select("title", "summary", "author","cleaned_title","cleaned_summary").limit(5).show())
        #print(df.drop("cleaned_title", "cleaned_summary", "sentiment_title", "sentiment_summary").show())
        #print(df.drop("cleaned_title", "cleaned_summary").show())
        #print(df.show())
        save_file_bigquery(merged_df,output_table)
        print(f"Data merged and saved to BigQuery table: {output_table}")
    else:
        print("No dataframes were merged.")



# Define PythonOperators
fetch_task = PythonOperator(
    task_id='fetch_papers',
    python_callable=fetch_data,
    dag=dag,
)

upload_task = PythonOperator(
    task_id='upload_to_gcs',
    python_callable=upload_data,
    dag=dag,
)

process_task = PythonOperator(
    task_id='process_and_load_bigquery',
    python_callable=transform_and_load,
    dag=dag,
)


# Define Task Dependencies
fetch_task >> upload_task >> process_task