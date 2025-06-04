from modules.fetching import fetch_papers_by_years
from modules.upload_file_gcs import create_bucket,upload_file_to_gcs
from modules.processing_spark import spark_init,set_spark_schema,process_column,calculate_sentiment,save_file_bigquery,spark_stop
from functools import reduce
from pathlib import Path
import argparse

# Data fetching, cleaning with pyspark and pushing to BigQuery
# Usage:
# If the bucket is not yet created
# python3 main.py --create-bucket
# If we already created it then
# python3 main.py --upload-gcs --process-spark --start_year 2000 --end_year 2025 --domain hep-ex

results_path = Path('data')
if not results_path.exists():
    results_path.mkdir()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Data fetching, cleaning with pyspark and pushing to BigQuery')

    # Boolean flags for data fetching and uploading to gcs and processing with PySpark and uploading to BigQyery
    parser.add_argument('--create-bucket', '--create_bucket',
                       action='store_true',
                       default=False,
                       help='Whether to create GCS bucket (default: False)')
    
    parser.add_argument('--upload-gcs', '--upload_gcs',
                action='store_true',
                default=False,
                help='Whether to upload local files into GCS bucket (default: False)')

    parser.add_argument('--process-spark', '--process_spark',
                action='store_true',
                default=False,
                help='Whether to process the files with spark and upload to BigQuery (default: False)')
    
    # Date range for data fetching
    parser.add_argument('--start-year', '--start_year',
                type=int,
                default=2000,
                help='Start year for feature creation (format: YYYY, default: 2000)')

    parser.add_argument('--end-year', '--end_year',
                    type=int,
                    default=2025,
                    help='End year for feature creation (format: YYYY, default: 2025)')
    
    # Which domain are we processing (hep-ex, cs, ...)
    parser.add_argument('--domain', '--domain',
            type=str,
            default='hep-ex',
            help='Domain for fetching and data pushing (default: hep-ex)')

    return parser.parse_args()


def main():

    # Parse arguments
    args = parse_arguments()

    # Access the variables
    CREATE_BUCKET = args.create_bucket
    UPLOAD_TO_GCS = args.upload_gcs
    PROCESS_SPARK = args.process_spark
    START_YEAR = args.start_year
    END_YEAR = args.end_year
    DOMAIN = args.domain


    # Print configuration
    print("Configuration:")
    print(f"CREATE_BUCKET = {CREATE_BUCKET}")
    print(f"UPLOAD_TO_GCS = {UPLOAD_TO_GCS}")
    print(f"PROCESS_SPARK = {PROCESS_SPARK}")
    print(f"START_YEAR = '{START_YEAR}'")
    print(f"END_YEAR = '{END_YEAR}'")
    print(f"DOMAIN = '{DOMAIN}'")

    # Download files locally
    for year in range(START_YEAR, END_YEAR, 1):
        df = fetch_papers_by_years(start_year=year, end_year=year+1, domain = DOMAIN, file_path="./data")
    
    # Upload files to GCS
    bucket_name = "arxiv-bucket-111865037319658112231"

    if CREATE_BUCKET:
        create_bucket(bucket_name)

    if UPLOAD_TO_GCS:
        for year in range(START_YEAR, END_YEAR, 1):
            local_file = f'./data/arxiv_{DOMAIN}_papers_{year}-{year+1}.parquet'
            destination_blob_name = f'arxiv_data/{DOMAIN}/arxiv_{DOMAIN}_papers_{year}-{year+1}.parquet'
            upload_file_to_gcs(bucket_name, local_file, destination_blob_name)


    # Process the GCS files using PySpark and then push them back BigQuery
    if PROCESS_SPARK:
        key_location = "./keys/arxiv-trends-key.json"
        spark = spark_init(key_file=key_location)


        dataframes = []
        for year in range(START_YEAR, END_YEAR, 1):
            gcs_file_name = f'arxiv_data/{DOMAIN}/arxiv_{DOMAIN}_papers_{year}-{year+1}.parquet'
            df = set_spark_schema(spark, file_name=f'gs://{bucket_name}/{gcs_file_name}')
            df = process_column(df, 'title')
            df = process_column(df, 'summary')
            df = calculate_sentiment(df, 'title')
            df = calculate_sentiment(df, 'summary')
            dataframes.append(df)


        # Merge the dataframes so we only push a single file per domain into BigQuery
        if dataframes:
            merged_df = reduce(lambda df1, df2: df1.unionByName(df2), dataframes)
        else:
            merged_df = None


        # save the file
        if merged_df:
            # BigQuery does not accept table names with dashes, have to replace them
            DOMAIN_CLEANED = DOMAIN.replace("-", "_")
            output_table = f'arxiv-trends.arxiv_papers.arxiv_{DOMAIN_CLEANED}_papers_{START_YEAR}_{END_YEAR}'
            save_file_bigquery(merged_df,output_table)
            print(f"Data merged and saved to BigQuery table: {output_table}")
        else:
            print("No dataframes were merged.")

        # stop Spark session
        spark_stop(spark)