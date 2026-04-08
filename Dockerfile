# Use lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Avoid Python buffering (clean logs)
ENV PYTHONUNBUFFERED=1

# 1. Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# 2. Install dependencies (this layer stays cached unless requirements.txt changes)
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the rest of the project files
COPY . .

# Default command
CMD ["python3", "inference.py"]