from google.cloud import bigquery


def migrate_to_bigquery(uri="gs://test-arxiv-bucket-111865037319658112231/arxiv_data/hep-ex/processed/*.parquet"):
    client = bigquery.Client(project='arxiv-trends')

    table_id = "arxiv-trends.arxiv_papers"

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE  # or WRITE_APPEND
    )


    load_job = client.load_table_from_uri(
        uri, table_id, job_config=job_config
    )

    load_job.result()  # Waits for the job to finish

    table = client.get_table(table_id)
    print(f"Loaded {table.num_rows} rows to {table_id}.")
