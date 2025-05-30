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
    
    print(f"DEBUG: Target directory for VACE-Annotators: {local_dir}")
    
    # Ensure the VACE-Annotators directory exists
    vace_annotators_target_dir = os.path.join(local_dir, 'VACE-Annotators')
    print(f"DEBUG: Ensuring directory: {vace_annotators_target_dir}")
    os.makedirs(vace_annotators_target_dir, exist_ok=True)
    
    # Check for key model files to see if already downloaded
    key_files = [
        os.path.join(vace_annotators_target_dir, 'depth', 'dpt_hybrid-midas-501f0c75.pt'),
        os.path.join(vace_annotators_target_dir, 'pose', 'yolox_l.onnx'),
        os.path.join(vace_annotators_target_dir, 'flow', 'raft-things.pth')
    ]
    
    models_exist = all(os.path.exists(f) for f in key_files)
    print(f"DEBUG: Key model files exist in {vace_annotators_target_dir}: {models_exist}")
    
    if not models_exist:
        print(f"VACE-Annotators models not found in {vace_annotators_target_dir}. Downloading...")
        # Download to a temporary directory first to avoid issues
        temp_download_dir = os.path.join(local_dir, "temp_vace_annotators_download")
        print(f"DEBUG: Downloading to temporary directory: {temp_download_dir}")
        os.makedirs(temp_download_dir, exist_ok=True)
        
        snapshot_download(
            repo_id="ali-vilab/VACE-Annotators",
            local_dir=temp_download_dir,
            local_dir_use_symlinks=False,
            # ignore_patterns=["*.md", ".git*"],  # Optional: ignore markdown and git files
        )
        print(f"Download complete to temporary directory: {temp_download_dir}")

        # The snapshot_download might create a VACE-Annotators subdir inside temp_download_dir
        # or download contents directly. We need to handle both.
        
        # Option 1: Contents are directly in temp_download_dir (e.g., depth/, pose/)
        # Option 2: Contents are in temp_download_dir/VACE-Annotators/ (e.g., VACE-Annotators/depth/)

        source_dir_option1 = temp_download_dir
        source_dir_option2 = os.path.join(temp_download_dir, "VACE-Annotators")

        actual_source_dir = None
        if os.path.exists(os.path.join(source_dir_option1, 'depth')):
             actual_source_dir = source_dir_option1
             print(f"DEBUG: Found models directly in {actual_source_dir}")
        elif os.path.exists(os.path.join(source_dir_option2, 'depth')):
            actual_source_dir = source_dir_option2
            print(f"DEBUG: Found models in subdirectory {actual_source_dir}")
        else:
            print(f"ERROR: Could not find downloaded model structure in {temp_download_dir}")
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_download_dir)
            return vace_annotators_target_dir # Return target, though download failed

        # Move contents from actual_source_dir to vace_annotators_target_dir
        # We need to be careful if vace_annotators_target_dir already has some files (e.g. from a partial previous download)
        import shutil
        print(f"DEBUG: Moving models from {actual_source_dir} to {vace_annotators_target_dir}")
        for item_name in os.listdir(actual_source_dir):
            s = os.path.join(actual_source_dir, item_name)
            d = os.path.join(vace_annotators_target_dir, item_name)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True) # Overwrite if exists
            else:
                shutil.copy2(s, d) # Overwrite if exists
        
        # Clean up temp directory
        shutil.rmtree(temp_download_dir)
        print(f"Models moved to {vace_annotators_target_dir}")
    else:
        print(f"VACE-Annotators models already exist at {vace_annotators_target_dir}")

    # Final confirmation of contents
    print(f"DEBUG: Final contents of {vace_annotators_target_dir}:")
    if os.path.exists(vace_annotators_target_dir):
        for item in os.listdir(vace_annotators_target_dir):
            item_path = os.path.join(vace_annotators_target_dir, item)
            if os.path.isdir(item_path):
                print(f"  DIR: {item}")
                try:
                    for subitem in os.listdir(item_path):
                        print(f"    {subitem}")
                except:
                    pass # Ignore permission errors for sub-listing
            else:
                print(f"  FILE: {item}")
    else:
        print(f"DEBUG: {vace_annotators_target_dir} does not exist after all operations.")

    return vace_annotators_target_dir 