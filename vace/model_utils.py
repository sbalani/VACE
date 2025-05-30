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

def ensure_annotator_models_downloaded(local_dir="models/VACE-Annotators"):
    """
    Ensures all annotator models are downloaded to the local directory.
    If the models already exist, they will not download again.
    
    Args:
        local_dir (str): Local directory to save the models to. This should be the final target, e.g., models/VACE-Annotators
        
    Returns:
        str: Path to the local model directory (e.g., /workspace/VACE/models/VACE-Annotators)
    """
    # Convert to absolute path if relative
    if not os.path.isabs(local_dir):
        # Check if we're in a workspace environment
        if os.path.exists('/workspace'):
            base_dir = '/workspace/VACE'
        else:
            base_dir = os.getcwd()
        final_target_dir = os.path.join(base_dir, local_dir) # e.g. /workspace/VACE/models/VACE-Annotators
    else:
        final_target_dir = local_dir

    print(f"DEBUG: Final target directory for VACE-Annotators: {final_target_dir}")
    os.makedirs(final_target_dir, exist_ok=True) # Ensure target directory exists

    # Check for key model files to see if already downloaded in the final target directory
    key_files = [
        os.path.join(final_target_dir, 'depth', 'dpt_hybrid-midas-501f0c75.pt'),
        os.path.join(final_target_dir, 'pose', 'yolox_l.onnx'),
        os.path.join(final_target_dir, 'flow', 'raft-things.pth')
    ]
    
    models_exist = all(os.path.exists(f) for f in key_files)
    print(f"DEBUG: Key model files exist in {final_target_dir}: {models_exist}")
    
    if not models_exist:
        # Download to a temporary directory to avoid messing with final_target_dir if it exists partially
        # The parent of final_target_dir is where we want snapshot_download to create its VACE-Annotators folder
        download_parent_dir = os.path.dirname(final_target_dir) # e.g. /workspace/VACE/models

        # Let snapshot_download create its own VACE-Annotators directory inside download_parent_dir
        print(f"VACE-Annotators models not found in {final_target_dir}. Downloading...")
        print(f"DEBUG: Calling snapshot_download to download into: {download_parent_dir}")
        
        snapshot_download(
            repo_id="ali-vilab/VACE-Annotators",
            local_dir=download_parent_dir, 
            local_dir_use_symlinks=False,
        )
        print(f"Download complete! snapshot_download placed files in {download_parent_dir}")

        # snapshot_download creates a subdirectory named after the repo, so models are in download_parent_dir/VACE-Annotators
        downloaded_vace_annotators_path = os.path.join(download_parent_dir, "VACE-Annotators")

        # If the final_target_dir is NOT the same as where snapshot_download put them, we might have a problem or a misunderstanding.
        # However, given our setup, final_target_dir (e.g., .../models/VACE-Annotators) 
        # IS EQUIVALENT to downloaded_vace_annotators_path (e.g., .../models/VACE-Annotators)
        # So, no move is strictly necessary if snapshot_download behaves as expected.
        # The key is that final_target_dir should now contain the models.

        if not os.path.exists(downloaded_vace_annotators_path):
             print(f"ERROR: snapshot_download did not create the expected {downloaded_vace_annotators_path} directory.")
             print(f"DEBUG: Contents of {download_parent_dir}:")
             if os.path.exists(download_parent_dir):
                for item in os.listdir(download_parent_dir):
                    print(f"  {item}")
             # No further action, will rely on the key_files check later
        else:
            # This check is mainly to ensure the logic is sound. If final_target_dir is indeed where snapshot_download placed it, this is redundant.
            if downloaded_vace_annotators_path != final_target_dir:
                print(f"WARNING: Download path {downloaded_vace_annotators_path} differs from target {final_target_dir}. This might indicate an issue.")
                # This would be the place to move files if snapshot_download behaved unexpectedly, 
                # but the expectation is that it places them in final_target_dir due to local_dir=download_parent_dir
                # and it creating a VACE-Annotators subdirectory.

        # Verify the download created the expected files in the final_target_dir
        if not all(os.path.exists(f) for f in key_files):
            print(f"ERROR: Download did not result in expected files in {final_target_dir} after download and processing.")
    else:
        print(f"VACE-Annotators models already exist at {final_target_dir}")

    # Final confirmation of contents
    print(f"DEBUG: Final contents of {final_target_dir}:")
    if os.path.exists(final_target_dir):
        for item in os.listdir(final_target_dir):
            item_path = os.path.join(final_target_dir, item)
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
        print(f"DEBUG: {final_target_dir} does not exist after all operations.")

    return final_target_dir

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