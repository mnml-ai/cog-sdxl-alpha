import torch
from PIL import Image
from ip_adapter import IPAdapterPlus

class IPAdapterHandler:
    def __init__(self, pipe):
        self.pipe = pipe
        self.ip_adapter = IPAdapterPlus(self.pipe, "./ip-adapter/ip-adapter_sdxl.bin", "./ip-adapter-plus/ip-adapter-plus_sdxl_vit-h.bin", "./ip-adapter-plus-face/ip-adapter-plus-face_sdxl_vit-h.bin")

    def preprocess_image(self, image_path, face=False):
        image = Image.open(image_path).convert("RGB")
        if face:
            image = self.ip_adapter.preprocess_face(image)
        return self.ip_adapter.preprocess(image)

    def generate_with_ip_adapter(self, prompt, image_path, num_samples=1, scale=1.0, face=False):
        image = self.preprocess_image(image_path, face)
        images = self.ip_adapter.generate(pil_image=image, num_samples=num_samples, num_inference_steps=30, seed=42, scale=scale, prompt=prompt)
        return images