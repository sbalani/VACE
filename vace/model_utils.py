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
    
    print(f"DEBUG: Checking for VACE-Annotators at: {local_dir}")
    print(f"DEBUG: Directory exists: {os.path.exists(local_dir)}")
    
    # Check for the actual VACE-Annotators subdirectory
    vace_annotators_dir = os.path.join(local_dir, 'VACE-Annotators')
    print(f"DEBUG: Checking for VACE-Annotators subdirectory at: {vace_annotators_dir}")
    print(f"DEBUG: VACE-Annotators subdirectory exists: {os.path.exists(vace_annotators_dir)}")
    
    if not os.path.exists(vace_annotators_dir):
        print(f"VACE-Annotators models not found at {vace_annotators_dir}. Downloading...")
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id="ali-vilab/VACE-Annotators",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"Download complete! Models saved to {local_dir}")
        
        # Debug: Show what's actually in the downloaded directory
        print(f"DEBUG: Contents of {local_dir}:")
        if os.path.exists(local_dir):
            for item in os.listdir(local_dir):
                item_path = os.path.join(local_dir, item)
                if os.path.isdir(item_path):
                    print(f"  DIR: {item}")
                    # Show subdirectories too
                    try:
                        for subitem in os.listdir(item_path):
                            print(f"    {subitem}")
                    except:
                        pass
                else:
                    print(f"  FILE: {item}")
    else:
        print(f"VACE-Annotators models already exist at {vace_annotators_dir}")
    
    return local_dir 