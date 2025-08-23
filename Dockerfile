FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Install any needed packages 
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Define the command to run your app when the container starts
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
