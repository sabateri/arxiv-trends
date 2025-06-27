provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_storage_bucket" "arxiv_data" {
  name     = var.bucket_name
  location = var.region
  force_destroy = true
  uniform_bucket_level_access = true
}

resource "google_bigquery_dataset" "arxiv_dataset" {
  dataset_id = var.dataset_id
  location   = var.region
  description = "Dataset for storing processed arXiv paper metadata"
}
