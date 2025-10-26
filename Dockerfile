# Use Python 3.12 slim image
FROM python:3.12-slim

# Set environment variable to disable Python output buffering
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies for Open3D and OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set the entrypoint to run main.py with unbuffered output
ENTRYPOINT ["python", "-u", "main.py"]

# Default arguments (can be overridden at runtime)
CMD ["--bag-dir", "/data/rosbag"]
