FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    jupyter \
    gradio \
    opencv-python \
    numpy \
    pillow \
    torchvision \
    transformers \
    accelerate \
    safetensors

# Create working directory
WORKDIR /workspace

# Copy the VACE code
COPY . /workspace/

# Create start script
COPY start.sh /workspace/
RUN chmod +x /workspace/start.sh

# Expose port for Jupyter
EXPOSE 8888

# Set the entrypoint
ENTRYPOINT ["/workspace/start.sh"] 