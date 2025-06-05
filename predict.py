# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import torch
import tempfile
import subprocess
from typing import List, Optional
from huggingface_hub import snapshot_download

# VACE specific imports
from vace.models.wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES
# Conditional import for preprocess tasks for choices, actual import for use later
VACE_PREPROCESS_TASK_CHOICES = sorted([
    'animate_anything', 'clipref', 'composition', 'depthv2', 'expand_anything', 
    'firstclip', 'firstframe', 'firstlastclip', 'firstlastframe', 'flow', 'frameref', 
    'framerefext', 'gray', 'image_depth', 'image_face', 'image_gray', 'image_inpainting', 
    'image_outpainting', 'image_plain', 'image_pose', 'image_reference', 'image_salient', 
    'image_scribble', 'inpainting', 'inpainting_bbox', 'inpainting_bboxtrack', 
    'inpainting_caption', 'inpainting_label', 'inpainting_mask', 'inpainting_masktrack', 
    'lastclip', 'lastframe', 'layout_bbox', 'layout_track', 'move_anything', 'outpainting', 
    'outpainting_inner', 'plain', 'pose', 'pose_body', 'reference_anything', 'scribble', 
    'swap_anything'
])

# Placeholder for vace_preproccess.parse_bboxes if needed directly
# For now, we assume that the string format is sufficient for the CLI call or direct function call
# If direct function call, this might be needed: from vace.vace_preproccess import parse_bboxes


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        os.makedirs("models", exist_ok=True)
        os.makedirs("models/VACE-Annotators", exist_ok=True)

        # Download VACE-Annotators models from Hugging Face
        print("Downloading VACE-Annotators models...")
        snapshot_download(
            repo_id="ali-vilab/VACE-Annotators",
            local_dir="models/VACE-Annotators",
            local_dir_use_symlinks=False
        )

        # Download model files (T5, VAE) - these are common
        model_files = {
            "models/models_t5_umt5-xxl-enc-bf16.pth": "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth",
            "models/Wan2.1_VAE.pth": "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth?download=true",
            "models/config.json": "https://huggingface.co/google/umt5-xxl/resolve/main/config.json",
            "models/special_tokens_map.json": "https://huggingface.co/google/umt5-xxl/resolve/main/special_tokens_map.json",
            "models/tokenizer_config.json": "https://huggingface.co/google/umt5-xxl/resolve/main/tokenizer_config.json",
            "models/spiece.model": "https://huggingface.co/google/umt5-xxl/resolve/main/spiece.model"
        }
        
        for filepath, url in model_files.items():
            if not os.path.exists(filepath):
                print(f"Downloading {os.path.basename(filepath)}...")
                subprocess.run(["wget", "-O", filepath, url], check=True)
        
        # Download 14B model shards (example, adjust if using 1.3B or other models primarily)
        # These are downloaded to models/Wan2.1-VACE-14B/
        # The ckpt_dir for inference will point to "models/Wan2.1-VACE-14B"
        # The self.vace_ckpt_name is relative to that directory.
        
        # Base model checkpoint directory and name for inference
        # This will be selected based on user input for model_name_input
        # self.ckpt_dir_base = "models" 
        # self.vace_ckpt_name = "model.safetensors.index.json" # for 14B, or .pth for 1.3B

        # Ensure VACE config paths are set correctly (already done by import)
        # Update config to use local tokenizer path, relative to where the script using it runs
        # vace_wan_inference.py and others might expect '.' to be <repo-root>/models/
        # The setup in cog.yaml usually makes the repo root the CWD.
        # If t5_tokenizer is expected to be 'models' by the underlying WanVace class:
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
        
        # Import main functions from VACE scripts AFTER potential sys.path modifications or library setup
        from vace.vace_preproccess import main as preprocess_main
        from vace.vace_wan_inference import main as inference_main
        self.preprocess_main = preprocess_main
        self.inference_main = inference_main


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
        
        # Inputs for vace_wan_inference.py (also used by pipeline if preprocess=False)
        input_video: Path = Input(description="Source video for video-to-video tasks or if preprocess=False.", default=None),
        input_mask: Path = Input(description="Source mask for inpainting/outpainting if preprocess=False or for direct inference.", default=None),
        input_ref_images: List[Path] = Input(description="Reference image(s) if preprocess=False or for direct inference. Comma-separated.", default=None),

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
        sample_steps: int = Input(description="Sampling steps.", default=None, ge=1, le=100), # Default handled by inference script
        sample_shift: float = Input(description="Noise schedule shift. Recommended 3.0 for 480p.", default=None), # Default handled by inference script
        sample_guide_scale: float = Input(description="Classifier free guidance scale.", default=5.0, ge=1.0, le=20.0),
        use_prompt_extend: str = Input(
            description="Prompt extension strategy.", 
            default='plain', 
            choices=['plain', 'wan_zh', 'wan_en', 'wan_zh_ds', 'wan_en_ds'] # from vace_wan_inference
        ),
        offload_model: Optional[bool] = Input(description="Offload model to CPU. Default: True for single GPU, False for multi.", default=None),
        # Multi-GPU args - typically not set by user in Cog, defaults in script are fine
        # ulysses_size: int = 1,
        # ring_size: int = 1,
        # t5_fsdp: bool = False,
        # t5_cpu: bool = False,
        # dit_fsdp: bool = False,

    ) -> Path:
        """Run a single prediction on the model"""

        temp_dir = tempfile.mkdtemp()
        
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
                preprocess_output = self.preprocess_main(preprocess_args_dict)
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
            "sample_steps": sample_steps, # None will use default in script
            "sample_shift": sample_shift, # None will use default
            "sample_guide_scale": sample_guide_scale,
            "use_prompt_extend": use_prompt_extend,
            "save_file": inference_save_path, # Direct output path
            "offload_model": offload_model if offload_model is not None else (torch.cuda.device_count() <= 1),
            # Other args like ulysses_size, ring_size, t5_fsdp will use defaults in vace_wan_inference.py
            # Ensure sample_fps is passed if vace_wan_inference.py uses it for saving (it uses cfg.sample_fps)
        }
        # Filter out None values for inference_args
        inference_args_dict = {k: v for k, v in inference_args.items()if v is not None}
        
        # Additional validation based on vace_wan_inference.py defaults if needed
        if 'sample_steps' not in inference_args_dict: # Set default based on inference script logic if desired
            # args.sample_steps = 50 (general default)
            inference_args_dict['sample_steps'] = 50 
        if 'sample_shift' not in inference_args_dict:
            # args.sample_shift = 16 (general default)
            inference_args_dict['sample_shift'] = 16


        print(f"Inference arguments: {inference_args_dict}")
        try:
            inference_output_data = self.inference_main(inference_args_dict)
            # inference_main saves the video to 'save_file' and returns dict with 'out_video' path
            if not inference_output_data or 'out_video' not in inference_output_data:
                if os.path.exists(inference_save_path): # Fallback if dict is empty but file was saved
                    print("Inference script did not return out_video path, but file exists at specified location.")
                else:
                    raise RuntimeError("Inference failed or did not produce an output video file.")
            
            output_path = Path(inference_output_data.get('out_video', inference_save_path))
            if not output_path.exists():
                 raise FileNotFoundError(f"Output video not found at {output_path}")
            print(f"Output video generated at: {output_path}")
            return output_path

        except Exception as e:
            print(f"Error during inference: {e}")
            # Consider cleaning up temp_dir in case of failure
            raise
        # finally:
            # shutil.rmtree(temp_dir) # Clean up temp files - Cog might handle this
