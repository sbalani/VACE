import os
from huggingface_hub import snapshot_download
import requests
from tqdm import tqdm

def ensure_model_downloaded(model_name="Wan-AI/Wan2.1-VACE-14B", local_dir="models/Wan2.1-VACE-14B"):
    """
    Returns the local directory path for the model.
    Model downloading functionality is disabled as models are now included in the Docker container.
    """
    # Convert to absolute path if relative
    if not os.path.isabs(local_dir):
        # Check if we're in a workspace environment
        if os.path.exists('/workspace'):
            base_dir = '/workspace/VACE'
        else:
            base_dir = os.getcwd()
        local_dir = os.path.join(base_dir, local_dir)
    
    return local_dir

def ensure_annotator_models_downloaded(local_dir_models_parent="models"):
    """
    Returns the path to the directory containing annotator model subdirectories.
    Model downloading functionality is disabled as models are now included in the Docker container.
    """
    # Resolve the absolute path for the directory that will contain /depth, /pose, etc.
    if not os.path.isabs(local_dir_models_parent):
        if os.path.exists('/workspace'):
            base_dir = '/workspace/VACE'
        else:
            base_dir = os.getcwd()
        actual_models_root_dir = os.path.join(base_dir, local_dir_models_parent) # e.g. /workspace/VACE/models
    else:
        actual_models_root_dir = local_dir_models_parent

    return actual_models_root_dir # This is the directory containing depth/, pose/, etc.

def debug_check_model_path(model_path: str, model_name: str = "Model"):
    """
    Prints the model path and checks if the file exists.
    """
    print(f"DEBUG_PATH_CHECK: {model_name} - Expected path: {model_path}")
    exists = os.path.exists(model_path)
    print(f"DEBUG_PATH_CHECK: {model_name} - File exists at path: {exists}")
    if not exists:
        # Try to list the parent directory to see what's there
        parent_dir = os.path.dirname(model_path)
        print(f"DEBUG_PATH_CHECK: {model_name} - Listing contents of parent directory ({parent_dir}):")
        if os.path.exists(parent_dir):
            try:
                for item in os.listdir(parent_dir):
                    print(f"  - {item}")
            except Exception as e:
                print(f"    Error listing parent directory: {e}")
        else:
            print(f"    Parent directory {parent_dir} does not exist.")
    return exists 