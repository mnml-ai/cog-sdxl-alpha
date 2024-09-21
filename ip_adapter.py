import os
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from ip_adapter import IPAdapterXL

class IPAdapter:
    def __init__(self, base_model_path, ip_adapter_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model_path = base_model_path
        self.ip_adapter_path = ip_adapter_path
        self.pipe = None
        self.ip_adapter = None

    def load_models(self):
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.base_model_path, 
            vae=vae,
            torch_dtype=torch.float16, 
            variant="fp16"
        ).to(self.device)

        self.ip_adapter = IPAdapterXL(self.pipe, self.ip_adapter_path, "cuda")

    def generate_image(self, prompt, image, num_samples=4, num_inference_steps=30, seed=None):
        if self.pipe is None or self.ip_adapter is None:
            self.load_models()

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        images = self.ip_adapter.generate(
            prompt=prompt,
            image=image,
            num_samples=num_samples,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        return images