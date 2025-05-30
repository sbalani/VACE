import os
from huggingface_hub import snapshot_download
import requests
from tqdm import tqdm

def ensure_model_downloaded(model_name="Wan-AI/Wan2.1-VACE-14B", local_dir="models/Wan2.1-VACE-14B"):
    """
    Ensures the model is downloaded to the local directory.
    If the model already exists, it will not download again.
    
    Args:
        model_name (str): The HuggingFace model name/ID
        local_dir (str): Local directory to save the model to
        
    Returns:
        str: Path to the local model directory
    """
    # Convert to absolute path if relative
    if not os.path.isabs(local_dir):
        # Check if we're in a workspace environment
        if os.path.exists('/workspace'):
            base_dir = '/workspace/VACE'
        else:
            base_dir = os.getcwd()
        local_dir = os.path.join(base_dir, local_dir)
    
    if not os.path.exists(local_dir):
        print(f"Model not found at {local_dir}. Downloading {model_name}...")
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"Download complete! Model saved to {local_dir}")
    else:
        print(f"Model already exists at {local_dir}")
    
    return local_dir

def ensure_annotator_models_downloaded(local_dir="models"):
    """
    Ensures all annotator models are downloaded to the local directory.
    If the models already exist, they will not download again.
    
    Args:
        local_dir (str): Local directory to save the models to
        
    Returns:
        str: Path to the local model directory
    """
    # Convert to absolute path if relative
    if not os.path.isabs(local_dir):
        # Check if we're in a workspace environment
        if os.path.exists('/workspace'):
            base_dir = '/workspace/VACE'
        else:
            base_dir = os.getcwd()
        local_dir = os.path.join(base_dir, local_dir)
    
    print(f"DEBUG: Checking for VACE-Annotators models at: {local_dir}")
    
    # Check for key model files to see if already downloaded
    key_files = [
        os.path.join(local_dir, 'depth', 'dpt_hybrid-midas-501f0c75.pt'),
        os.path.join(local_dir, 'pose', 'yolox_l.onnx'),
        os.path.join(local_dir, 'flow', 'raft-things.pth')
    ]
    
    models_exist = all(os.path.exists(f) for f in key_files)
    print(f"DEBUG: Key model files exist: {models_exist}")
    
    if not models_exist:
        print(f"VACE-Annotators models not found. Downloading...")
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id="ali-vilab/VACE-Annotators",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"Download complete! Models saved to {local_dir}")
    else:
        print(f"VACE-Annotators models already exist at {local_dir}")
    
    return local_dir 