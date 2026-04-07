# Use lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Avoid Python buffering (clean logs)
ENV PYTHONUNBUFFERED=1

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["python3", "evaluate_agents.py"]