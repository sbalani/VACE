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

def ensure_annotator_models_downloaded(local_dir_models_parent="models"):
    """
    Ensures all annotator models from ali-vilab/VACE-Annotators are downloaded.
    The contents of the VACE-Annotators repo will be placed directly into the 'models' directory.
    
    Args:
        local_dir_models_parent (str): The directory where model subdirectories (depth, pose, etc.) will reside.
                                     Typically "models".
        
    Returns:
        str: Absolute path to the directory containing model subdirectories (e.g., /workspace/VACE/models)
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

    print(f"DEBUG: Root directory for annotator model subdirectories (depth, pose, etc.): {actual_models_root_dir}")
    os.makedirs(actual_models_root_dir, exist_ok=True)

    # Check for key model files to see if they are already downloaded directly in actual_models_root_dir
    key_files = [
        os.path.join(actual_models_root_dir, 'depth', 'dpt_hybrid-midas-501f0c75.pt'),
        os.path.join(actual_models_root_dir, 'pose', 'yolox_l.onnx'),
        os.path.join(actual_models_root_dir, 'flow', 'raft-things.pth')
    ]
    
    models_exist = all(os.path.exists(f) for f in key_files)
    print(f"DEBUG: Key model files exist in {actual_models_root_dir}: {models_exist}")
    
    if not models_exist:
        print(f"VACE-Annotators models not found in {actual_models_root_dir}. Downloading...")
        # snapshot_download will place contents of VACE-Annotators repo directly into actual_models_root_dir
        print(f"DEBUG: Calling snapshot_download to download VACE-Annotators repo contents into: {actual_models_root_dir}")
        
        snapshot_download(
            repo_id="ali-vilab/VACE-Annotators",
            local_dir=actual_models_root_dir, 
            local_dir_use_symlinks=False,
            # ignore_patterns=["*.md", ".git*"], # Example: ignore non-model files
        )
        print(f"Download complete! snapshot_download placed VACE-Annotators repo contents into {actual_models_root_dir}")

        # Verify the download created the expected files
        if not all(os.path.exists(f) for f in key_files):
            print(f"ERROR: Download did not result in expected files in {actual_models_root_dir}.")
            print(f"DEBUG: Contents of {actual_models_root_dir} after download attempt:")
            if os.path.exists(actual_models_root_dir):
                for item in os.listdir(actual_models_root_dir):
                    print(f"  {item}")
    else:
        print(f"VACE-Annotators models (contents) already exist in {actual_models_root_dir}")

    # Final confirmation of contents
    print(f"DEBUG: Final contents of {actual_models_root_dir}:")
    if os.path.exists(actual_models_root_dir):
        for item in os.listdir(actual_models_root_dir):
            item_path = os.path.join(actual_models_root_dir, item)
            if os.path.isdir(item_path):
                print(f"  DIR: {item}")
                try:
                    for subitem in os.listdir(item_path):
                        print(f"    {subitem}")
                except:
                    pass 
            else:
                print(f"  FILE: {item}")
    else:
        print(f"DEBUG: {actual_models_root_dir} does not exist after all operations.")

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