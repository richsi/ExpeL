# Choose a CUDA 12.6 base image
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu20.04

# Set environment variables

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3.9 \
  python3.9-venv \
  python3-pip \
  build-essential \
  git \
  curl \
  libjpeg-dev \
  libpng-dev \
  libopencv-dev \
  && rm -rf /var/lib/apt/lists/*

# Optionally set python3.9 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Set working directory
WORKDIR /app

# Copy the entire repository into the container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r new_requirements.txt



CMD ["python", "train.py", "benchmark=hotpotqa", "run_name=docker_train_mistral7b_3", "testing=false", "resume=false"]