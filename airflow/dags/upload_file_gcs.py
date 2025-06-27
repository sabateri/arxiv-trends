from google.cloud import storage
import os


def create_bucket(bucket_name):
    """
    Create a new bucket in the EU region
    """
    storage_client = storage.Client(project='arxiv-trends')

    bucket = storage_client.bucket(bucket_name)
    new_bucket = storage_client.create_bucket(bucket, location="europe-west6")

    print(
        "Created bucket {} in {}".format(
            new_bucket.name, new_bucket.location
        )
    )
    return new_bucket




def upload_file_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """
    Uploads a local file to a GCS bucket.
    """
    storage_client = storage.Client(project='arxiv-trends')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    #bucket = storage_client.create_bucket(bucket_name)
    print(f"File {source_file_name} uploaded to gs://{bucket_name}/{destination_blob_name}.")

# example usage:
# bucket_name = "test-arxiv-bucket-111865037319658112231"
# local_file = "/home/sabateri/code/projects/arxiv_wordFreq/data/arxiv_hep-ex_papers_2010_2024.parquet"
# destination_blob_name = "arxiv_data/arxiv_hep-ex_papers_2012_2018.parquet"

# CREATE_BUCKET = False
# if CREATE_BUCKET:
#     create_bucket(bucket_name)
# upload_file_to_gcs(bucket_name, local_file, destination_blob_name)

