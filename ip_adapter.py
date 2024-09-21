import os
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from huggingface_hub import hf_hub_download

class IPAdapter:
    def __init__(self, pipe: StableDiffusionXLPipeline, repo_id: str, cache_dir: str):
        self.pipe = pipe
        self.device = pipe.device
        self.dtype = pipe.dtype
        
        # Download and load the IP-Adapter image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=self.dtype
        ).to(self.device)
        
        self.feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # Download IP-Adapter weights
        ip_adapter_path = hf_hub_download(repo_id=repo_id, 
                                          filename="sdxl_models/ip-adapter-plus_sdxl_vit-h.bin", 
                                          cache_dir=cache_dir)
        
        self.ip_layers = torch.load(ip_adapter_path, map_location="cpu")
        self.ip_layers = {k: v.to(self.device, dtype=self.dtype) for k, v in self.ip_layers.items()}

    def preprocess_image(self, image):
        if isinstance(image, str):
            image = Image.open(image)
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values
        image = image.to(self.device, dtype=self.dtype)
        return image

    def apply_to_pipeline(self, pipe):
        def attn_processor(attn, hidden_states, encoder_hidden_states, attention_mask=None):
            batch_size, sequence_length, _ = hidden_states.shape
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            query = attn.to_q(hidden_states)
            
            is_cross_attention = encoder_hidden_states is not None
            if is_cross_attention:
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)
                
                # IP-Adapter cross-attention forward
                ip_key = self.ip_layers["key"]
                ip_value = self.ip_layers["value"]
                
                key = torch.cat([key, ip_key], dim=1)
                value = torch.cat([value, ip_value], dim=1)
            else:
                key = attn.to_k(hidden_states)
                value = attn.to_v(hidden_states)
            
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            
            return hidden_states

        # Apply the custom attention processor to all cross-attention layers
        for module in pipe.unet.modules():
            if module.__class__.__name__ == "Attention" and module.is_cross_attention:
                module.processor = attn_processor

    def unapply_from_pipeline(self, pipe):
        # Restore the original attention processors
        for module in pipe.unet.modules():
            if module.__class__.__name__ == "Attention" and module.is_cross_attention:
                module.processor = None

    def unload(self):
        del self.image_encoder
        del self.ip_layers
        torch.cuda.empty_cache()

    def __call__(self, attn, hidden_states, encoder_hidden_states, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            
            # IP-Adapter cross-attention forward
            ip_key = self.ip_layers["key"]
            ip_value = self.ip_layers["value"]
            
            key = torch.cat([key, ip_key], dim=1)
            value = torch.cat([value, ip_value], dim=1)
        else:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states