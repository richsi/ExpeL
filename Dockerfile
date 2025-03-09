# Choose a CUDA 12.6 base image
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04

# Set environment variables for non-interactive installs and timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/PST

# Pre-configure tzdata to avoid interactive prompts
RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3 \
  python3-venv \
  python3-pip \
  python3-dev \
  build-essential \
  git \
  curl \
  python-is-python3 \
  rustc \
  cargo \
  && rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /app

# Copy the entire repository into the container
COPY . /app

# Install Python dependencies
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/python -m pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt \
 && /opt/venv/bin/pip install -r new_requirements.txt

ENV PATH="/opt/venv/bin:$PATH"

# Run the training script
CMD ["python", "train.py", "benchmark=hotpotqa", "run_name=docker_train_mistral7b_3", "testing=false", "resume=false"]
