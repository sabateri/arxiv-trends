FROM apache/airflow:3.0.2

# Set working directory in the container
WORKDIR /app

# Switch to root for installing system dependencies
USER root
# Install Java
#RUN apt-get update && apt-get install -y openjdk-11-jdk
#RUN apt-get update && apt install -y default-jre && apt-get clean
RUN apt-get update && apt-get install -y default-jdk && apt-get clean

# Set JAVA_HOME environment variable
# Check where JAVA is by running the container and then $ update-alternatives --config java
#ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Get wget
RUN apt install wget

# Create spark jars directory
RUN mkdir -p /opt/spark/jars

# Download BigQuery connector
RUN wget -O /opt/spark/jars/spark-bigquery-with-dependencies.jar \
    "https://repo1.maven.org/maven2/com/google/cloud/spark/spark-bigquery-with-dependencies_2.12/0.37.0/spark-bigquery-with-dependencies_2.12-0.37.0.jar"

# Download GCS connector
RUN wget -O /opt/spark/jars/gcs-connector-hadoop3-latest.jar \
    "https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar"


# Switch back to airflow user for security
USER airflow

# Copy requirements file first
COPY requirements.txt /app/

# Install dependencies

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir arxiv pandas nltk pyarrow google-cloud-storage
RUN pip install --no-cache-dir pyspark==3.5.4 textblob pandas nltk

# RUN mkdir -p /opt/spark/jars && \
#     curl -L -o /opt/spark/jars/gcs-connector.jar https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar


# Copy the rest of the application
COPY . /app/