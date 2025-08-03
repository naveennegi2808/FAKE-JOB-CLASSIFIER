# Start with a standard Python 3.11 base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Update the system, install build tools and the fortran compiler
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app's code
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# The command to run your app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]