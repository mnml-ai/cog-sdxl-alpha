import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

class IPAdapterPlus:
    def __init__(self, pipe, ip_ckpt, clip_vision_model="h94/IP-Adapter", subfolder="sdxl_models", dtype=torch.float16):
        self.pipe = pipe
        self.dtype = dtype
        self.device = torch.device("cuda")
        
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_vision_model, subfolder=subfolder).to(self.device, dtype)
        self.feature_extractor = CLIPImageProcessor.from_pretrained(clip_vision_model, subfolder=subfolder)
        
        # Custom implementation of ImageProjection
        self.image_proj_model = nn.Linear(self.image_encoder.config.hidden_size, pipe.unet.config.cross_attention_dim).to(self.device, dtype)
        
        self.load_ip_adapter(ip_ckpt)
        
    def load_ip_adapter(self, ckpt):
        state_dict = torch.load(ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])
        
    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, nn.Module):
                attn_processor.scale = scale
            
    def generate(self, ip_adapter_image, **kwargs):
        image = ip_adapter_image.resize((224, 224))
        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device, self.dtype)
        image_embeds = self.image_encoder(**inputs).image_embeds
        image_prompt_embeds = self.image_proj_model(image_embeds)
        
        uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds)
        kwargs["ip_adapter_image_embeds"] = torch.cat([uncond_image_prompt_embeds, image_prompt_embeds])
        
        return self.pipe(**kwargs)