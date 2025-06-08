import os
import sys
import tempfile
from typing import List

# Add project root to python path for vace imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # Add current directory
sys.path.insert(0, os.path.join(current_dir, "vace"))  # Add vace directory

from cog import BasePredictor, Input, Path
from huggingface_hub import hf_hub_download, snapshot_download

# Import the vace_wan_inference function
from vace.vace_wan_inference import main as vace_wan_inference
from vace.models.wan import WanVace


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)

        # Download the full Wan2.1-VACE-14B repository
        print("Downloading Wan2.1-VACE-14B repository...")
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-VACE-14B",
            local_dir="models",
            local_dir_use_symlinks=False
        )

        # Download model files if they don't exist, using huggingface_hub
        model_files = {
            "special_tokens_map.json": {
                "repo_id": "google/umt5-xxl",
                "filename": "special_tokens_map.json"
            },
            "tokenizer_config.json": {
                "repo_id": "google/umt5-xxl",
                "filename": "tokenizer_config.json"
            },
            "spiece.model": {
                "repo_id": "google/umt5-xxl",
                "filename": "spiece.model"
            }
        }

        for filename, info in model_files.items():
            filepath = os.path.join("models", filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename} from {info['repo_id']}...")
                hf_hub_download(
                    repo_id=info["repo_id"],
                    filename=info["filename"],
                    local_dir="models",
                    local_dir_use_symlinks=False
                )

        # Model paths are relative to the project root
        self.ckpt_dir = "models"

    def predict(
        self,
        model_name: str = Input(
            description="The model name to run.",
            choices=["vace-1.3B", "vace-14B"],
            default="vace-14B",
        ),
        prompt: str = Input(
            description="The prompt to generate the video from.", default="A woman walking down the street"
        ),
        src_video: Path = Input(
            description="The source video file.", default=None
        ),
        src_mask: Path = Input(
            description="The source mask file.", default=None
        ),
        src_ref_images: List[Path] = Input(
            description="List of source reference images.", default=None
        ),
        size: str = Input(
            description="The resolution of the generated video. The aspect ratio of the output video will follow the input.",
            default="480p",
            choices=["480p", "720p", "1080p"],
        ),
        frame_num: int = Input(
            description="Number of frames to sample. Must be 4n+1.", default=81
        ),
        base_seed: int = Input(
            description="Seed for generation. -1 for random.", default=-1
        ),
        sample_solver: str = Input(
            description="Solver for sampling.",
            choices=["unipc", "dpm++"],
            default="unipc",
        ),
        sample_steps: int = Input(
            description="Sampling steps. Defaults to 50 for T2V and 40 for I2V.",
            default=25,
        ),
        sample_shift: float = Input(
            description="Shift factor for flow matching schedulers.", default=5.0
        ),
        sample_guide_scale: float = Input(
            description="Classifier-free guidance scale.", default=5.0
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        # Prepare arguments
        args = type('Args', (), {
            'model_name': model_name,
            'prompt': prompt,
            'src_video': str(src_video) if src_video else None,
            'src_mask': str(src_mask) if src_mask else None,
            'src_ref_images': [str(p) for p in src_ref_images] if src_ref_images else None,
            'size': size,
            'frame_num': frame_num,
            'base_seed': base_seed,
            'sample_solver': sample_solver,
            'sample_steps': sample_steps,
            'sample_shift': sample_shift,
            'sample_guide_scale': sample_guide_scale,
            'ckpt_dir': self.ckpt_dir,
            'offload_model': False,
            'ulysses_size': 1,
            'ring_size': 1,
            't5_fsdp': False,
            't5_cpu': False,
            'dit_fsdp': False,
            'use_prompt_extend': 'plain',
            'save_file': None,
        })()

        # Create a temporary directory for the output
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "output.mp4")
        args.save_file = output_path

        # Generate output using vace_wan_inference
        vace_wan_inference(args)

        # Return the output path
        return Path(output_path) 