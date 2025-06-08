import os
import sys
import tempfile
import subprocess
import torch
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
        # Print CUDA and NCCL version information
        print("CUDA Version:", torch.version.cuda)
        print("NCCL Version:", torch.cuda.nccl.version())
        print("PyTorch Version:", torch.__version__)
        print("CUDA Available:", torch.cuda.is_available())
        print("CUDA Device Count:", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("CUDA Device Name:", torch.cuda.get_device_name(0))
            print("CUDA Device Capability:", torch.cuda.get_device_capability(0))

        # Create a temporary directory for the output
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "output.mp4")

        # Prepare arguments for vace_wan_inference
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
            'ulysses_size': 2,  # Number of GPUs
            'ring_size': 1,     # For 14B model
            't5_fsdp': True,    # Enable FSDP for T5
            't5_cpu': False,
            'dit_fsdp': True,   # Enable FSDP for DIT
            'use_prompt_extend': 'plain',
            'save_file': output_path,
        })()

        # Run vace_wan_inference with torch.distributed.run for multi-GPU support
        env = os.environ.copy()
        env.update({
            "NCCL_DEBUG": "INFO",
            "NCCL_IGNORE_CPU_AFFINITY": "1",
            "NCCL_IB_DISABLE": "1",
            "NCCL_P2P_DISABLE": "1",
            "NCCL_SOCKET_IFNAME": "lo",
            "NCCL_IB_TIMEOUT": "1800",
            "NCCL_DEBUG_SUBSYS": "ALL",
            "TORCH_NCCL_BLOCKING_WAIT": "1",
            "NCCL_ASYNC_ERROR_HANDLING": "1"
        })

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "vace/vace_wan_inference.py",
            "--model_name", args.model_name,
            "--prompt", args.prompt,
            "--ckpt_dir", args.ckpt_dir,
            "--size", args.size,
            "--frame_num", str(args.frame_num),
            "--base_seed", str(args.base_seed),
            "--sample_solver", args.sample_solver,
            "--sample_steps", str(args.sample_steps),
            "--sample_shift", str(args.sample_shift),
            "--sample_guide_scale", str(args.sample_guide_scale),
            "--ulysses_size", str(args.ulysses_size),
            "--ring_size", str(args.ring_size),
            "--t5_fsdp",
            "--dit_fsdp",
            "--use_prompt_extend", args.use_prompt_extend,
            "--save_file", args.save_file
        ]

        if args.src_video:
            cmd.extend(["--src_video", args.src_video])
        if args.src_mask:
            cmd.extend(["--src_mask", args.src_mask])
        if args.src_ref_images:
            cmd.extend(["--src_ref_images"] + args.src_ref_images)

        # Run the command with the modified environment
        subprocess.run(cmd, check=True, env=env)

        # Return the output path
        return Path(output_path) 