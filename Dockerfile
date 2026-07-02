# Dockerfile
# This Dockerfile defines a container image for our Python services (FastAPI and Streamlit).
# A Docker image is a packaged environment that contains our code, dependencies, and configuration.
# This ensures that the code runs identically on any computer, server, or cloud VM without
# manual setup issues (like "it works on my machine").

# =========================================================================================
# Advanced Concept: Dockerization and Containerized Environments
# A container isolates execution environments. We start from a base image (Python 3.10),
# install system dependencies (like g++ for compilation of certain Python extensions),
# copy our code, install our requirements, and expose the ports needed to communicate
# with the outside world.
# =========================================================================================

# Use the official Python 3.10 slim image as our starting base.
# Slim images are lightweight because they omit unnecessary Unix packages.
FROM python:3.10-slim

# Set the working directory inside the container's virtual filesystem.
# All subsequent instructions (like COPY or RUN) will execute relative to this folder.
WORKDIR /app

# Install system dependencies required by libraries like lxml, PyTorch, or sentence-transformers.
# We run apt-get update to fetch package indices, and install gcc/g++ compilers.
# We clean up apt cache afterwards to keep the image size small.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file from the host project directory into the container.
COPY requirements.txt .

# Install the Python dependencies listed in requirements.txt.
# We use --no-cache-dir to avoid storing downloaded packages inside the image, reducing size.
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files and folders from our project directory into the container's /app directory.
COPY . .

# Expose ports 8000 (FastAPI API) and 8501 (Streamlit Web UI).
# Exposing tells Docker that the container will listen on these network ports at runtime.
EXPOSE 8000
EXPOSE 8501

# The default command if no specific command is provided when running the container.
# We default to spinning up the FastAPI backend web server using Uvicorn.
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
