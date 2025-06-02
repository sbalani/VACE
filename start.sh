#!/bin/bash

export DEBIAN_FRONTEND=noninteractive
export HF_HOME="/workspace"

start_jupyter(){
    echo "Starting Jupyter Notebook..."
    mkdir -p /workspace/logs
    cd / && \
    nohup jupyter lab --allow-root \
    --no-browser \
    --port 8888 \
    --ip 0.0.0.0 \
    --FileContentsManager.delete_to_trash=False \
    --ContentsManager.allow_hidden=True \
    --SeverApp.terminado_settings='{"shell_command":["/bin/bash"]}' \
    --ServerApp.token='' \
    --ServerApp.allow_origin='*' \
    --SeverApp.preferred_dir='/workspace' &> /workspace/logs/jupyter.log &
    echo "Jupyter Notebook started. Access it at http://localhost:8888"
}

start_jupyter


set -e  # Exit on error

echo "Starting VACE setup..."

# Show current directory contents
echo "Current directory contents:"
pwd
ls -la

# If workspace is empty, clone the repo
if [ -z "$(ls -A /workspace)" ]; then
    echo "Workspace is empty, cloning VACE repository..."
    cd /workspace
    git clone -b feature/combined-ui https://github.com/sbalani/VACE.git
    cd VACE
else
    # Try to find and cd to VACE directory
    cd /workspace/VACE
fi

echo -e "\nCurrent directory contents:"
ls -la

# Install Python dependencies
echo -e "\nInstalling Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Error: requirements.txt not found!"
    exit 1
fi

# Install additional required packages
echo "Installing Wan and xfuser packages..."
pip install wan@git+https://github.com/Wan-Video/Wan2.1
pip install "xfuser>=0.4.1"

# Create models directory if it doesn't exist
mkdir -p models

# Download models using huggingface-cli
echo "Downloading models..."
echo "Downloading VACE-Annotators..."
huggingface-cli download ali-vilab/VACE-Annotators --local-dir models/VACE-Annotators

echo "Downloading Wan2.1-VACE-14B..."
huggingface-cli download Wan-AI/Wan2.1-VACE-14B --local-dir models/Wan2.1-VACE-14B

echo "Setup complete! Models have been downloaded to:"
echo "- models/Wan2.1-VACE-14B"
echo "- models/VACE-Annotators"
echo ""
echo "You can now start the VACE application using:"
echo "python vace/gradios/vace_wan_complete.py --model_name vace-14B --ckpt_dir models/Wan2.1-VACE-14B --server_name \"0.0.0.0\" --server_port 7860 --share"

# Start Jupyter Notebook instead of Lab


# Start VACE application
echo -e "\nStarting VACE application..."
python vace/gradios/vace_wan_complete.py --model_name vace-14B --ckpt_dir models/Wan2.1-VACE-14B --server_name "0.0.0.0" --server_port 7860 --share