# Use Python 3.11 as the base image
FROM python:3.11-slim

# Set working directory in the container
WORKDIR /app


# Copy requirements file first (if you have one)
# This is a good practice to leverage Docker cache
COPY requirements.txt /app/

# Install dependencies
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . /app/

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
#CMD ["python", "app.py"]
CMD ["gunicorn", "app:app", "--timeout", "300", "--bind", "0.0.0.0:5000"]



