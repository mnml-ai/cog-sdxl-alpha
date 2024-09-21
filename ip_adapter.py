import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline
from ip_adapter import IPAdapterPlusXL

class IPAdapterPredictor:
    def __init__(self):
        # Define paths
        self.IP_ADAPTER_IMAGE_ENCODER_PATH = "/IP-Adapter/models/image_encoder"
        self.IP_ADAPTER_SDXL_PATH = "/IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"

        # Initialize the IP-Adapter model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ip_adapter = IPAdapterPlusXL(
            img2img_pipe=self.load_img2img_pipeline(),
            encoder_path=self.IP_ADAPTER_IMAGE_ENCODER_PATH,
            model_path=self.IP_ADAPTER_SDXL_PATH,
            device=self.device,
            num_tokens=16,
        )

    def load_img2img_pipeline(self):
        # Load the necessary SDXL img2img pipeline
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "./sdxl-cache",  # Assuming the SDXL model is cached in the same path as your original script
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        return pipe.to(self.device)

    def generate_images(self, input_image_path, prompt, negative_prompt="", num_outputs=1, steps=30, scale=7.5, adapter_scale=0.7):
        # Load the input image
        ip_image = Image.open(input_image_path).convert("RGB")
        ip_image = ip_image.resize((224, 224))

        # Generate images using IP-Adapter
        images = self.ip_adapter.generate(
            pil_image=ip_image,
            num_samples=num_outputs,
            num_inference_steps=steps,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=scale,
            scale=adapter_scale,
        )

        # Return generated images
        return images
