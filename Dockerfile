FROM docker.io/runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies with retry logic and mirror selection
RUN for i in $(seq 1 3); do \
        sed -i 's/archive.ubuntu.com/mirrors.edge.kernel.org/g' /etc/apt/sources.list \
        && apt-get update -y \
        && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            git \
            ffmpeg \
            libsm6 \
            libxext6 \
        && rm -rf /var/lib/apt/lists/* \
        && break \
        || { echo "Retrying apt-get install... (attempt $i/3)"; sleep 2; }; \
    done

WORKDIR /workspace

# Copy the entire application at once
COPY . /workspace/VACE/

# Debug: Show contents
RUN echo "VACE directory contents:" && ls -la /workspace/VACE/

# Copy start script
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"] 