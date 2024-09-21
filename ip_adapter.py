import os
import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image
from weights_downloader import WeightsDownloader

class IPAdapter:
    def __init__(self, pipe, repo, cache_dir):
        self.pipe = pipe
        self.repo = repo
        self.cache_dir = cache_dir
        self.ip_adapter = None
        self.image_encoder = None
        self.image_processor = None
        
    def download_weights(self):
        WeightsDownloader.download_if_not_exists(self.repo, self.cache_dir)
        
    def load(self):
        if self.ip_adapter is None:
            self.download_weights()
            self.ip_adapter = torch.load(os.path.join(self.cache_dir, "ip_adapter_sdxl.bin"), map_location="cuda")
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=torch.float16
            ).to("cuda")
            self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
    def unload(self):
        del self.ip_adapter
        del self.image_encoder
        del self.image_processor
        torch.cuda.empty_cache()
        self.ip_adapter = None
        self.image_encoder = None
        self.image_processor = None
        
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.image_processor(image, return_tensors="pt").pixel_values
        return image.to("cuda", dtype=torch.float16)
        
    def apply_to_pipeline(self, pipe):
        self.load()
        pipe.unet = self.ip_adapter.adapt_unet(pipe.unet)
        pipe.image_encoder = self.image_encoder
        
    def unapply_from_pipeline(self, pipe):
        pipe.unet = self.ip_adapter.unadapt_unet(pipe.unet)
        pipe.image_encoder = None