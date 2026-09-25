# Use a stable NVIDIA CUDA runtime image. 
# 'runtime' is much smaller than 'devel' and includes everything PyTorch needs.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Install Python, pip, and required system packages
# libgl1 and libglib2.0-0 are required by OpenCV (used by Ultralytics)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory within the container
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install Python packages specified in requirements.txt.
# Ultralytics will automatically pull down PyTorch as a dependency.
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy the entire src directory into the container
COPY src /app/src

# Set the entrypoint to run the training script as a module.
ENTRYPOINT ["python3", "-m", "src.train"]
