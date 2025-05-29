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
            base_dir = '/workspace'
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

def ensure_midas_model_downloaded(local_dir="models/VACE-Annotators/depth"):
    """
    Ensures the MiDaS model is downloaded to the local directory.
    If the model already exists, it will not download again.
    
    Args:
        local_dir (str): Local directory to save the model to
        
    Returns:
        str: Path to the local model file
    """
    # Convert to absolute path if relative
    if not os.path.isabs(local_dir):
        # Check if we're in a workspace environment
        if os.path.exists('/workspace'):
            base_dir = '/workspace'
        else:
            base_dir = os.getcwd()
        local_dir = os.path.join(base_dir, local_dir)
    
    model_file = os.path.join(local_dir, "dpt_hybrid-midas-501f0c75.pt")
    
    if not os.path.exists(model_file):
        print(f"MiDaS model not found at {model_file}. Downloading...")
        os.makedirs(local_dir, exist_ok=True)
        
        url = "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt"
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_file, 'wb') as f, tqdm(
            desc="Downloading MiDaS model",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        
        print(f"Download complete! Model saved to {model_file}")
    else:
        print(f"MiDaS model already exists at {model_file}")
    
    return model_file

def ensure_annotator_models_downloaded(local_dir="models/VACE-Annotators"):
    """
    Ensures all annotator models are downloaded to the local directory.
    If the models already exist, they will not download again.
    
    Args:
        local_dir (str): Local directory to save the models to
        
    Returns:
        dict: Paths to the local model files
    """
    # Convert to absolute path if relative
    if not os.path.isabs(local_dir):
        # Check if we're in a workspace environment
        if os.path.exists('/workspace'):
            base_dir = '/workspace'
        else:
            base_dir = os.getcwd()
        local_dir = os.path.join(base_dir, local_dir)
    
    # Create base directory
    os.makedirs(local_dir, exist_ok=True)
    
    # Define model URLs and their local paths
    models = {
        'dwpose_det': {
            'url': 'https://huggingface.co/camenduru/unianimate/resolve/main/yolox_l.onnx?download=true',
            'path': os.path.join(local_dir, 'dwpose', 'yolox_l.onnx')
        },
        'dwpose_pose': {
            'url': 'https://huggingface.co/camenduru/unianimate/resolve/main/dw-ll_ucoco_384.onnx?download=true',
            'path': os.path.join(local_dir, 'dwpose', 'dw-ll_ucoco_384.onnx')
        },
        'salient': {
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
            'path': os.path.join(local_dir, 'salient', 'sam_vit_h_4b8939.pth')
        },
        'flow': {
            'url': 'https://huggingface.co/sbalani/raft-things/resolve/main/raft-things.pth?download=true',
            'path': os.path.join(local_dir, 'flow', 'raft-things.pth')
        }
    }
    
    # Download each model if it doesn't exist
    for model_name, model_info in models.items():
        if not os.path.exists(model_info['path']):
            print(f"{model_name} model not found at {model_info['path']}. Downloading...")
            os.makedirs(os.path.dirname(model_info['path']), exist_ok=True)
            
            response = requests.get(model_info['url'], stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_info['path'], 'wb') as f, tqdm(
                desc=f"Downloading {model_name} model",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
            
            print(f"Download complete! Model saved to {model_info['path']}")
        else:
            print(f"{model_name} model already exists at {model_info['path']}")
    
    return {name: info['path'] for name, info in models.items()} 