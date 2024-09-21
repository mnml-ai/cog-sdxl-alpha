import os
import torch
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image
from weights_downloader import WeightsDownloader
from safetensors.torch import load_file

class IPAdapter:
    def __init__(self, pipe, repo, cache_dir):
        self.pipe = pipe
        self.repo = repo
        self.cache_dir = cache_dir
        self.ip_adapter = None
        self.image_encoder = None
        self.image_processor = None
        self.is_loaded = False
        
    def download_weights(self):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            local_path = os.path.join(self.cache_dir, "ip-adapter-plus_sdxl_vit-h.safetensors")
            WeightsDownloader.download_if_not_exists(self.repo, local_path)
            return True
        except Exception as e:
            print(f"Error downloading IP Adapter weights: {str(e)}")
            return False
        
    def load(self):
        if self.is_loaded:
            return True
        
        if not self.download_weights():
            return False
        
        try:
            local_path = os.path.join(self.cache_dir, "ip-adapter-plus_sdxl_vit-h.safetensors")
            self.ip_adapter = load_file(local_path)
            
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=torch.float16
            ).to("cuda")
            self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Error loading IP Adapter components: {str(e)}")
            return False
        
    def unload(self):
        if hasattr(self.pipe, 'orig_unet'):
            self.pipe.unet = self.pipe.orig_unet
        del self.ip_adapter
        del self.image_encoder
        del self.image_processor
        torch.cuda.empty_cache()
        self.is_loaded = False
        
    def preprocess_image(self, image_path):
        if not self.is_loaded:
            if not self.load():
                return None
        image = Image.open(image_path).convert("RGB")
        image = self.image_processor(images=image, return_tensors="pt").pixel_values
        return image.to("cuda", dtype=torch.float16)
        
    def apply_to_pipeline(self, pipe):
        if not self.is_loaded:
            if not self.load():
                return False
        if not hasattr(pipe, 'orig_unet'):
            pipe.orig_unet = pipe.unet
        
        # Apply the IP-Adapter weights to the UNet
        attn_procs = {}
        for key in self.ip_adapter:
            if '.alpha' in key:
                attn_procs[key] = self.ip_adapter[key].item()
            else:
                attn_procs[key] = self.ip_adapter[key].to(device=pipe.device, dtype=pipe.dtype)
        pipe.unet.set_attn_processor(attn_procs)
        return True
        
    def unapply_from_pipeline(self, pipe):
        if hasattr(pipe, 'orig_unet'):
            pipe.unet = pipe.orig_unet
        
    def encode_image(self, image):
        if not self.is_loaded:
            if not self.load():
                return None
        return self.image_encoder(image).image_embeds