variable "project_id" {
  type        = string
  description = "GCP Project ID"
}

variable "region" {
  type        = string
  default     = "europe-west6"
  description = "Region for GCS and BQ"
}

variable "bucket_name" {
  type        = string
  description = "Name of the GCS bucket"
}

variable "dataset_id" {
  type        = string
  description = "BigQuery dataset ID"
}
