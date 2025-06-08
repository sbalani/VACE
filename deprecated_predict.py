# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import torch
import tempfile
import subprocess
import json
import shutil
import sys
from typing import Optional
from huggingface_hub import snapshot_download, hf_hub_download

# VACE specific imports
from vace.models.wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES
# Conditional import for preprocess tasks for choices, actual import for use later
VACE_PREPROCESS_TASK_CHOICES = sorted([
    # Image preprocessing tasks
    'image_plain', 'image_face', 'image_salient', 'image_inpainting', 'image_reference', 
    'image_outpainting', 'image_depth', 'image_gray', 'image_pose', 'image_scribble',
    
    # Video preprocessing tasks  
    'plain', 'depth', 'depthv2', 'flow', 'gray', 'pose', 'pose_body', 'scribble',
    'framerefext', 'frameref', 'clipref', 'firstframe', 'lastframe', 'firstlastframe',
    'firstclip', 'lastclip', 'firstlastclip', 'inpainting', 'inpainting_mask', 
    'inpainting_bbox', 'inpainting_masktrack', 'inpainting_bboxtrack', 'inpainting_label', 
    'inpainting_caption', 'outpainting', 'outpainting_inner', 'layout_bbox', 'layout_track',
    
    # Composition preprocessing tasks
    'composition', 'reference_anything', 'animate_anything', 'swap_anything', 
    'expand_anything', 'move_anything'
])

# Placeholder for vace_preproccess.parse_bboxes if needed directly
# For now, we assume that the string format is sufficient for the CLI call or direct function call
# If direct function call, this might be needed: from vace.vace_preproccess import parse_bboxes


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_gpus = torch.cuda.device_count()
        print(f"Detected {self.num_gpus} GPUs")
        
        os.makedirs("models", exist_ok=True)
        os.makedirs("models/VACE-Annotators", exist_ok=True)

        # Try to add swap space (may fail on Replicate, but worth trying)
        try:
            print("Attempting to add swap space...")
            subprocess.run(["sudo", "fallocate", "-l", "4G", "/tmp/swapfile"], check=False, capture_output=True)
            subprocess.run(["sudo", "chmod", "600", "/tmp/swapfile"], check=False, capture_output=True)
            subprocess.run(["sudo", "mkswap", "/tmp/swapfile"], check=False, capture_output=True)
            result = subprocess.run(["sudo", "swapon", "/tmp/swapfile"], check=False, capture_output=True)
            if result.returncode == 0:
                print("Successfully added 4GB swap space")
            else:
                print("Could not add swap space (likely running in restricted environment)")
        except Exception as e:
            print(f"Swap space setup failed (expected in containerized environments): {e}")

        # Download VACE-Annotators models from Hugging Face
        print("Downloading VACE-Annotators models...")
        snapshot_download(
            repo_id="ali-vilab/VACE-Annotators",
            local_dir="models/VACE-Annotators",
            local_dir_use_symlinks=False
        )

        # Add VACE-Annotators to Python path for annotators import
        annotators_path = os.path.abspath("models/VACE-Annotators")
        if annotators_path not in sys.path:
            sys.path.insert(0, annotators_path)

        # Add current directory to path and establish package hierarchy for imports
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        # Establish proper package hierarchy and models alias
        try:
            # First import vace package to establish proper package hierarchy
            import vace
            import vace.models
            import vace.models.wan
            sys.modules['models'] = vace.models
            print("Added 'models' as alias for 'vace.models'")
        except Exception as e:
            print(f"Could not set up models alias: {e}")

        # Download UMT5-XXL tokenizer files
        print("Downloading UMT5-XXL tokenizer files...")
        snapshot_download(
            repo_id="google/umt5-xxl",
            local_dir="models",
            local_dir_use_symlinks=False,
            allow_patterns=["config.json", "special_tokens_map.json", "tokenizer_config.json", "spiece.model"]
        )

        # Download Wan model files
        print("Downloading Wan model files...")
        wan_files = [
            "models_t5_umt5-xxl-enc-bf16.pth",
            "Wan2.1_VAE.pth"
        ]
        
        for filename in wan_files:
            local_path = f"models/{filename}"
            if not os.path.exists(local_path):
                print(f"Downloading {filename}...")
                hf_hub_download(
                    repo_id="Wan-AI/Wan2.1-T2V-1.3B",
                    filename=filename,
                    local_dir="models",
                    local_dir_use_symlinks=False
                )

        # Install xfuser for multi-GPU acceleration if not already installed
        try:
            import xfuser
            print("xfuser already installed")
        except ImportError:
            print("Installing xfuser for multi-GPU support...")
            env = os.environ.copy()
            env["MAX_JOBS"] = "1"  # Limit compilation jobs
            subprocess.run(["pip", "install", "xfuser>=0.4.1"], check=True, env=env)

        # Update config to use local tokenizer path
        for config_key in list(WAN_CONFIGS.keys()):
            config = WAN_CONFIGS[config_key]
            if hasattr(config, 't5_tokenizer'):
                if isinstance(config, dict):
                    config['t5_tokenizer'] = "models" 
                else:
                    try:
                        config.t5_tokenizer = "models"
                    except AttributeError:
                         print(f"Warning: Could not set t5_tokenizer for {config_key}.")

    def _get_distributed_config(self, model_name: str, num_gpus: int):
        """Get the appropriate distributed configuration based on model and available GPUs"""
        if model_name == "vace-1.3B":
            # For 1.3B model: --ulysses_size 1 --ring_size N
            return {
                "ulysses_size": 1,
                "ring_size": num_gpus,
                "dit_fsdp": True,
                "t5_fsdp": True
            }
        elif model_name == "vace-14B":
            # For 14B model: --ulysses_size N --ring_size 1
            return {
                "ulysses_size": num_gpus,
                "ring_size": 1, 
                "dit_fsdp": True,
                "t5_fsdp": True
            }
        else:
            # Default configuration
            return {
                "ulysses_size": 1,
                "ring_size": num_gpus,
                "dit_fsdp": True,
                "t5_fsdp": True
            }

    def _run_distributed_inference(self, inference_args_dict: dict, num_gpus: int) -> str:
        """Run inference using torchrun for distributed execution"""
        
        # Create a temporary script to run the inference
        temp_script_path = "/tmp/run_inference.py"
        
        # Create the inference script content
        script_content = f'''
import sys
import os
import json

# Add current directory to path to import vace modules
sys.path.insert(0, os.getcwd())

from vace.vace_wan_inference import main as inference_main

# Load arguments from environment or file
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--args_file", required=True)
args = parser.parse_args()

with open(args.args_file, 'r') as f:
    inference_args = json.load(f)

# Run inference
try:
    result = inference_main(inference_args)
    print(f"Inference completed successfully: {{result}}")
except Exception as e:
    print(f"Inference failed: {{e}}")
    raise
'''
        
        # Write the script
        with open(temp_script_path, 'w') as f:
            f.write(script_content)
        
        # Save inference arguments to a JSON file
        args_file = "/tmp/inference_args.json"
        with open(args_file, 'w') as f:
            json.dump(inference_args_dict, f, indent=2)
        
        # Build the torchrun command
        torchrun_cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            temp_script_path,
            f"--args_file={args_file}"
        ]
        
        print(f"Running distributed inference with command: {' '.join(torchrun_cmd)}")
        
        try:
            # Run the distributed inference
            result = subprocess.run(
                torchrun_cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=os.getcwd()
            )
            
            print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
                
            # Return the output video path from inference_args_dict
            return inference_args_dict.get("save_file", "output.mp4")
            
        except subprocess.CalledProcessError as e:
            print(f"Distributed inference failed with return code {e.returncode}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            raise RuntimeError(f"Distributed inference failed: {e.stderr}")
        finally:
            # Clean up temporary files
            if os.path.exists(temp_script_path):
                os.remove(temp_script_path)
            if os.path.exists(args_file):
                os.remove(args_file)

    def predict(
        self,
        model_name_input: str = Input(
            description="The VACE base model to use.",
            default="vace-14B",
            choices=["vace-1.3B", "vace-14B"] # Corresponds to keys in WAN_CONFIGS
        ),
        prompt: str = Input(description="The prompt to generate from"),
        size: str = Input(
            description="The size of the output (e.g., '720p' for 1280x720, '480p' for 832x480, or 'W*H')",
            default="720p",
            choices=["1280*720", "832*480", "480*832", "480*480", "720p", "480p"] 
        ),
        frame_num: int = Input(
            description="Number of frames to generate (4n+1)",
            default=81, ge=1, le=121
        ),
        sample_fps: int = Input(
            description="FPS of the generated video",
            default=24, ge=1, le=60
        ),
        base_seed: int = Input(description="Random seed for generation", default=-1),
        
        # Multi-GPU Configuration
        use_multi_gpu: bool = Input(
            description="Use multi-GPU distributed inference (recommended for memory issues)",
            default=True
        ),
        num_gpus_override: Optional[int] = Input(
            description="Override number of GPUs to use (default: use all available)",
            default=None, ge=1, le=8
        ),
        
        # Inputs for vace_wan_inference.py (also used by pipeline if preprocess=False)
        input_video: Path = Input(description="Source video for video-to-video tasks or if preprocess=False.", default=None),
        input_mask: Path = Input(description="Source mask for inpainting/outpainting if preprocess=False or for direct inference.", default=None),
        input_ref_images: list[Path] = Input(description="Reference image(s) if preprocess=False or for direct inference. Comma-separated.", default=None),

        # Preprocessing control
        run_preprocess: bool = Input(description="Run VACE preprocessing pipeline before inference.", default=False),
        pipeline_task: str = Input(
            description="The VACE preprocessing task to run if 'run_preprocess' is true.",
            choices=VACE_PREPROCESS_TASK_CHOICES,
            default="depth"
        ),
        # Common Preprocessing arguments from vace_preproccess.py
        # These are passed to vace_preproccess.py if run_preprocess is True
        preprocess_image_input: Path = Input(description="Image input for image-based preprocessing tasks.", default=None),
        preprocess_video_input: Path = Input(description="Video input for video-based preprocessing tasks (can be same as input_video).", default=None),
        preprocess_mask_input: Path = Input(description="Mask input for mask-based preprocessing tasks (can be same as input_mask).", default=None),
        preprocess_mode: Optional[str] = Input(description="Mode for specific preprocessing tasks (e.g., 'firstframe', 'bboxtrack').", default=None),
        preprocess_bbox: Optional[str] = Input(description="Bounding box for preprocessing (e.g., 'x1,y1,x2,y2 x1,y1,x2,y2').", default=None), # String format for multiple bboxes
        preprocess_label: Optional[str] = Input(description="Label(s) for preprocessing, comma-separated.", default=None),
        preprocess_caption: Optional[str] = Input(description="Caption for preprocessing.", default=None),
        preprocess_direction: Optional[str] = Input(description="Direction for outpainting (e.g., 'left,right'), comma-separated.", default=None),
        preprocess_expand_ratio: Optional[float] = Input(description="Expand ratio for outpainting/mask augmentation.", default=None),
        preprocess_expand_num: Optional[int] = Input(description="Number of frames for extension tasks.", default=None),
        preprocess_maskaug_mode: Optional[str] = Input(description="Mode for mask augmentation.", default=None),
        preprocess_maskaug_ratio: Optional[float] = Input(description="Ratio for mask augmentation.", default=None),

        # Inference specific parameters from vace_wan_inference.py
        sample_solver: str = Input(description="Solver for sampling.", default="unipc", choices=["unipc", "dpm++"]),
        sample_steps: int = Input(description="Sampling steps.", default=25, ge=1, le=100),
        sample_shift: float = Input(description="Noise schedule shift. Recommended 3.0 for 480p.", default=5.0),
        sample_guide_scale: float = Input(description="Classifier free guidance scale.", default=5.0, ge=1.0, le=20.0),
        use_prompt_extend: str = Input(
            description="Prompt extension strategy.", 
            default='plain', 
            choices=['plain', 'wan_zh', 'wan_en', 'wan_zh_ds', 'wan_en_ds'] # from vace_wan_inference
        ),

    ) -> Path:
        """Run a single prediction on the model"""

        temp_dir = tempfile.mkdtemp()
        
        # Determine number of GPUs to use
        effective_num_gpus = num_gpus_override if num_gpus_override else self.num_gpus
        if not use_multi_gpu:
            effective_num_gpus = 1
        
        print(f"Using {effective_num_gpus} GPUs for inference")
        
        # --- Argument mapping for VACE scripts ---
        # Model name for WAN_CONFIGS and inference script
        model_name_for_scripts = model_name_input
        if model_name_for_scripts not in WAN_CONFIGS:
            raise ValueError(f"Unsupported model_name_input: {model_name_for_scripts}. Available: {list(WAN_CONFIGS.keys())}")

        # Size mapping (Cog input "W*H" or "Xp" to VACE's "Xp" like keys in SIZE_CONFIGS)
        vace_size_key = size
        if "*" in size: # e.g. "1280*720"
            w, h = map(int, size.split('*'))
            # Attempt to find a matching key in SIZE_CONFIGS
            # This is a heuristic. VACE's SIZE_CONFIGS keys are like '720p', '480p'.
            # And SUPPORTED_SIZES[model_name_for_scripts] uses these keys.
            if w == 1280 and h == 720: vace_size_key = "720p"
            elif w == 720 and h == 1280: vace_size_key = "720p_vert" # Assuming a key like this might exist or be added
            elif w == 832 and h == 480: vace_size_key = "480p"
            elif w == 480 and h == 832: vace_size_key = "480p_vert" # Assuming
            elif w == 480 and h == 480: vace_size_key = "480p_sq" # Assuming
            # If no direct mapping, the validation later might catch it or SIZE_CONFIGS might handle W*H strings.
            # For safety, we should ensure vace_size_key is one of the supported keys.
            if vace_size_key not in SUPPORTED_SIZES.get(model_name_for_scripts, []):
                 # Try to find a key in SIZE_CONFIGS that matches the area or dimensions
                found_key = None
                for key, dims in SIZE_CONFIGS.items():
                    if isinstance(dims, (list, tuple)) and len(dims) == 2:
                        if (dims[0] == h and dims[1] == w): # VACE stores (height, width)
                            if key in SUPPORTED_SIZES.get(model_name_for_scripts, []):
                                found_key = key
                                break
                if found_key:
                    vace_size_key = found_key
                else:
                    raise ValueError(f"Unsupported size string '{size}' for model '{model_name_for_scripts}'. Could not map to a supported VACE size like '720p'. Supported: {SUPPORTED_SIZES.get(model_name_for_scripts, [])}")
        
        if model_name_for_scripts not in SUPPORTED_SIZES or vace_size_key not in SUPPORTED_SIZES[model_name_for_scripts]:
            supported_for_model = SUPPORTED_SIZES.get(model_name_for_scripts, [])
            raise ValueError(f"Unsupported size '{vace_size_key}' for model '{model_name_for_scripts}'. Supported sizes: {list(supported_for_model)}")

        # Determine checkpoint directory based on model name
        # Assuming models are downloaded into subdirectories named after the model, e.g., "models/Wan2.1-VACE-14B"
        # And the actual .pth or .safetensors file is inside that.
        # vace_wan_inference.py expects ckpt_dir to be the *directory*
        
        # Default convention from VACE README and inference scripts:
        # models/VACE-Wan2.1-1.3B-Preview/ or models/Wan2.1-VACE-14B/
        # Let's try to deduce this.
        ckpt_dir_for_inference = f"models/Wan2.1-{model_name_input.upper()}" # e.g. models/Wan2.1-VACE-14B
        if model_name_input == "vace-1.3B": # From README, one variant is VACE-Wan2.1-1.3B-Preview
            ckpt_dir_for_inference = "models/VACE-Wan2.1-1.3B-Preview" # Or adjust if a different 1.3B is primary
        
        # Ensure the specific model directory exists for downloads if we automate it here
        # For now, assume user has placed models correctly as per VACE instructions or they are downloaded by setup.
        # Example: Check for specific file if we knew the exact name, e.g. for 14B it's model.safetensors.index.json
        # For 1.3B, it might be a .pth file. The inference script handles loading from ckpt_dir.

        # --- Preprocessing Step ---
        processed_src_video = str(input_video) if input_video else None
        processed_src_mask = str(input_mask) if input_mask else None
        processed_src_ref_images_list = [str(p) for p in input_ref_images] if input_ref_images else None
        processed_src_ref_images_str = ",".join(processed_src_ref_images_list) if processed_src_ref_images_list else None

        if run_preprocess:
            print(f"Running preprocessing task: {pipeline_task}")
            preprocess_args = {
                "task": pipeline_task,
                "video": str(preprocess_video_input) if preprocess_video_input else (str(input_video) if input_video else None),
                "image": str(preprocess_image_input) if preprocess_image_input else None, # Can also take from input_ref_images
                "mask": str(preprocess_mask_input) if preprocess_mask_input else (str(input_mask) if input_mask else None),
                "mode": preprocess_mode,
                "bbox": preprocess_bbox, # Already string format, parse_bboxes is in vace_preproccess.py
                "label": preprocess_label,
                "caption": preprocess_caption,
                "direction": preprocess_direction,
                "expand_ratio": preprocess_expand_ratio,
                "expand_num": preprocess_expand_num,
                "maskaug_mode": preprocess_maskaug_mode,
                "maskaug_ratio": preprocess_maskaug_ratio,
                "pre_save_dir": os.path.join(temp_dir, "preprocessed"), # Save intermediate to temp
                "save_fps": sample_fps # Use same FPS for consistency
            }
            # Filter out None values for preprocess_args
            preprocess_args_dict = {k: v for k, v in preprocess_args.items() if v is not None}
            
            # Handle image list for preprocess_image_input if pipeline_task expects 'images'
            if preprocess_image_input and pipeline_task in VACE_PREPROCESS_TASK_CHOICES: # Simplified check
                 # Some tasks (e.g. comp_refany_anno) expect 'images' as a list of paths in config,
                 # but vace_preproccess.py takes a comma-separated string for --image if multiple.
                 # If preprocess_image_input is used, ensure it's a single path or handle accordingly
                 # For now, assuming single path for preprocess_image_input.
                 # If input_ref_images are to be used for a preprocess task needing multiple images:
                if not preprocess_args_dict.get("image") and processed_src_ref_images_str:
                    # Example: if task is 'reference_anything' and preprocess_image_input not set
                    if pipeline_task == 'reference_anything': # This is a composition task
                         preprocess_args_dict["image"] = processed_src_ref_images_str


            print(f"Preprocessing arguments: {preprocess_args_dict}")
            try:
                # Import main functions from VACE scripts AFTER potential sys.path modifications or library setup
                from vace.vace_preproccess import main as preprocess_main
                preprocess_output = preprocess_main(preprocess_args_dict)
                print(f"Preprocessing output: {preprocess_output}")
                # Update inference inputs with preprocessed paths
                processed_src_video = preprocess_output.get("src_video", processed_src_video)
                processed_src_mask = preprocess_output.get("src_mask", processed_src_mask)
                # src_ref_images from preprocess_output is a comma-separated string or None
                processed_src_ref_images_str = preprocess_output.get("src_ref_images", processed_src_ref_images_str)

            except Exception as e:
                print(f"Error during preprocessing: {e}")
                raise

        # --- Inference Step ---
        output_video_filename = "final_output.mp4"
        inference_save_path = os.path.join(temp_dir, output_video_filename)

        # Get distributed configuration
        dist_config = self._get_distributed_config(model_name_for_scripts, effective_num_gpus)

        inference_args = {
            "model_name": model_name_for_scripts,
            "size": vace_size_key,
            "frame_num": frame_num,
            "ckpt_dir": ckpt_dir_for_inference, # This should be like "models/Wan2.1-VACE-14B"
            "prompt": prompt,
            "base_seed": base_seed,
            "src_video": processed_src_video,
            "src_mask": processed_src_mask,
            "src_ref_images": processed_src_ref_images_str, # Comma-separated string
            "sample_solver": sample_solver,
            "sample_steps": sample_steps,
            "sample_shift": sample_shift,
            "sample_guide_scale": sample_guide_scale,
            "use_prompt_extend": use_prompt_extend,
            "save_file": inference_save_path, # Direct output path
            "offload_model": False if use_multi_gpu else True, # Disable offload for multi-GPU
            # Add distributed configuration
            **dist_config
        }
        # Filter out None values for inference_args
        inference_args_dict = {k: v for k, v in inference_args.items() if v is not None}
        
        if use_multi_gpu and effective_num_gpus > 1:
            print(f"Running distributed inference across {effective_num_gpus} GPUs")
            output_path_str = self._run_distributed_inference(inference_args_dict, effective_num_gpus)
        else:
            print("Running single-GPU inference")
            # Fallback to single GPU inference - import and call directly
            from vace.vace_wan_inference import main as inference_main
            inference_output_data = inference_main(inference_args_dict)
            output_path_str = inference_output_data.get('out_video', inference_save_path)
        
        output_path = Path(output_path_str)
        if not output_path.exists():
             raise FileNotFoundError(f"Output video not found at {output_path}")
        print(f"Output video generated at: {output_path}")
        return output_path