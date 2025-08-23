# Use an official Python runtime as a parent image
# Using 'slim' for a smaller image size
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# --- THIS IS THE NEW LINE ---
# Install system dependencies needed for building Python packages like faiss-cpu
RUN apt-get update && apt-get install -y build-essential

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Define the command to run your app when the container starts
# --server.address=0.0.0.0 allows the app to be accessible from outside the container
# --server.port=8501 specifies the port
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
