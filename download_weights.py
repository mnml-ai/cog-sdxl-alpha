#!/usr/bin/env python

import os
import sys
import torch
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download

# Append project directory to path so predict.py can be imported
sys.path.append('.')

# Import constants from predict.py
from predict import (
    SDXL_MODEL_CACHE,
    SDXL_URL,
    IP_ADAPTER_REPO,
    IP_ADAPTER_CACHE
)

# Ensure cache directories exist
os.makedirs(SDXL_MODEL_CACHE, exist_ok=True)
os.makedirs(IP_ADAPTER_CACHE, exist_ok=True)

print("Downloading SDXL base model...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    SDXL_URL,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    cache_dir=SDXL_MODEL_CACHE,
)
print("SDXL base model downloaded and cached.")

print("Downloading IP-Adapter weights...")
ip_adapter_path = hf_hub_download(
    repo_id=IP_ADAPTER_REPO,
    filename="sdxl_models/ip-adapter-plus_sdxl_vit-h.bin",
    cache_dir=IP_ADAPTER_CACHE
)
print(f"IP-Adapter weights downloaded and cached at {ip_adapter_path}")

print("Downloading IP-Adapter image encoder...")
hf_hub_download(
    repo_id="openai/clip-vit-large-patch14",
    filename="pytorch_model.bin",
    cache_dir=os.path.join(IP_ADAPTER_CACHE, "clip_vit_large_patch14")
)
print("IP-Adapter image encoder downloaded and cached.")

print("All models and weights have been successfully downloaded and cached.")