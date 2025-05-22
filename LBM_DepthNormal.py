import os
import torch
from tqdm import tqdm
import requests
import shutil

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file

from .lbm.models.lbm import LBMModel, LBMConfig
from .lbm.models.unets import DiffusersUNet2DCondWrapper
from .lbm.models.vae import AutoencoderKLDiffusers
from .lbm.models.embedders import ConditionerWrapper
from diffusers.models import AutoencoderKL
from diffusers import FlowMatchEulerDiscreteScheduler

class LBM_DepthNormal:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "task": (["depth", "normal"], {"default": "depth", "tooltip": "Select task type"}),
                "image": ("IMAGE",),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100, "tooltip": "Sampling steps"}),
                "precision": (["fp32", "bf16", "fp16"], {"default": "bf16", "tooltip": "Inference precision"}),
            },
            "optional": {
                "bridge_noise_sigma": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.1, "step": 0.001, "tooltip": "Controls diversity"}),
                "mask": ("MASK", {"tooltip": "Optional mask"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = '🧪AILab/🔆LBM'

    def process(self, task, image, steps, precision, bridge_noise_sigma=0.1, mask=None):
        model_map = {
            "depth": "LBM_depth.safetensors",
            "normal": "LBM_normals.safetensors"
        }
        model_url_map = {
            "depth": "https://huggingface.co/jasperai/LBM_depth/resolve/main/model.safetensors",
            "normal": "https://huggingface.co/jasperai/LBM_normals/resolve/main/model.safetensors"
        }
        model_name = model_map[task]
        model_url = model_url_map[task]
        model_path = self.ensure_model_exists(model_name, model_url)

        dtype_map = {
            "bf16": torch.bfloat16, 
            "fp16": torch.float16, 
            "fp32": torch.float32
        }
        base_dtype = dtype_map[precision]
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        lbm_model = self.create_lbm_model(base_dtype, bridge_noise_sigma, task)
        sd = load_torch_file(model_path, device=offload_device, safe_load=True)
        param_count = sum(1 for _ in lbm_model.named_parameters())
        for name, param in tqdm(lbm_model.named_parameters(),
                desc=f"Loading model parameters",
                total=param_count,
                leave=True):
            if name in sd:
                param.data = sd[name].to(dtype=base_dtype)
        mm.soft_empty_cache()

        input_image = image.clone().permute(0, 3, 1, 2).to(device, base_dtype) * 2 - 1
        batch = {"source_image": input_image}

        # Mask support
        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(0)
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            mask_tensor = mask.to(device, base_dtype)
            batch["mask"] = mask_tensor

        lbm_model.vae.to(device)
        z_source = lbm_model.vae.encode(batch[lbm_model.source_key])
        lbm_model.vae.cpu()
        lbm_model.to(device)

        result = lbm_model.sample(
            z=z_source,
            num_steps=steps,
            conditioner_inputs=batch,
            max_samples=1,
        ).clamp(-1, 1)

        out = result.permute(0, 2, 3, 1).cpu().float()
        out = (out + 1) / 2

        if task == "depth":
            out = 1 - out

        lbm_model.cpu()
        mm.soft_empty_cache()
        return (out,)

    def ensure_model_exists(self, model_name, model_url):
        model_paths = folder_paths.get_folder_paths("diffusion_models")
        if not model_paths:
            raise RuntimeError("No diffusion_models paths found")
        for path in model_paths:
            # Add LBM subfolder
            lbm_path = os.path.join(path, "LBM")
            model_path = os.path.join(lbm_path, model_name)
            if os.path.exists(model_path):
                print(f"Model {model_name} found at {model_path}")
                return model_path
        download_path = os.path.join(model_paths[0], "LBM")
        print(f"Model {model_name} not found. Downloading to {download_path}...")
        os.makedirs(download_path, exist_ok=True)
        target_path = os.path.join(download_path, model_name)
        temp_file = os.path.join(download_path, "temp_download.safetensors")
        try:
            with requests.get(model_url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                with open(temp_file, 'wb') as f, tqdm(
                    desc=f"Downloading {model_name}",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            shutil.move(temp_file, target_path)
            print(f"Model downloaded and saved as {target_path}")
            return target_path
        except Exception as e:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            print(f"Error downloading model: {e}")
            raise RuntimeError(f"Failed to download model: {e}")

    def create_lbm_model(self, dtype, bridge_noise_sigma=0.1, task="depth"):
        # Task specific configurations
        task_configs = {
            "depth": {
                "prob": [0.025, 0.05, 0.025, 0.9],
                "target_key": "depth"
            },
            "normal": {
                "prob": [0.05, 0.1, 0.05, 0.8],
                "target_key": "normals"
            }
        }
        
        task_config = task_configs[task]
        config = {
            "source_key": "source_image",
            "target_key": task_config["target_key"],
            "timestep_sampling": "custom_timesteps",
            "selected_timesteps": [250, 500, 750, 1000],
            "prob": task_config["prob"],
            "bridge_noise_sigma": bridge_noise_sigma,
        }
        denoiser = DiffusersUNet2DCondWrapper(
            in_channels=4,
            out_channels=4,
            center_input_sample=False,
            flip_sin_to_cos=True,
            freq_shift=0,
            down_block_types=[
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
            ],
            mid_block_type="UNetMidBlock2DCrossAttn",
            up_block_types=["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
            only_cross_attention=False,
            block_out_channels=[320, 640, 1280],
            layers_per_block=2,
            downsample_padding=1,
            mid_block_scale_factor=1,
            dropout=0.0,
            act_fn="silu",
            norm_num_groups=32,
            norm_eps=1e-05,
            cross_attention_dim=[320, 640, 1280],
            transformer_layers_per_block=[1, 2, 10],
            attention_head_dim=[5, 10, 20],
            use_linear_projection=True,
            time_embedding_type="positional",
        ).to(dtype)
        conditioner = ConditionerWrapper(conditioners=[])
        vae_config = {
            "_class_name": "AutoencoderKL",
            "_diffusers_version": "0.20.0.dev0",
            "act_fn": "silu",
            "block_out_channels": [128, 256, 512, 512],
            "down_block_types": [
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D"
            ],
            "force_upcast": True,
            "in_channels": 3,
            "latent_channels": 4,
            "layers_per_block": 2,
            "norm_num_groups": 32,
            "out_channels": 3,
            "sample_size": 1024,
            "scaling_factor": 0.13025,
            "up_block_types": [
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D"
            ]
        }
        vae = AutoencoderKLDiffusers(AutoencoderKL.from_config(vae_config))
        vae.freeze()
        vae.to(dtype)
        scheduler_config = {
            'num_train_timesteps': 1000, 
            'shift': 1.0, 
            'use_dynamic_shifting': False, 
            'beta_schedule': 'scaled_linear', 
            'beta_start': 0.00085, 
            'beta_end': 0.012, 
            'timestep_spacing': 'leading',
        }
        sampling_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        lbm_config = LBMConfig(**config)
        model = LBMModel(
            lbm_config,
            denoiser=denoiser,
            sampling_noise_scheduler=sampling_noise_scheduler,
            vae=vae,
            conditioner=conditioner,
        ).to(dtype)
        return model

NODE_CLASS_MAPPINGS = {
    "LBM_DepthNormal": LBM_DepthNormal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LBM_DepthNormal": "Depth / Normal (LBM)",
} 