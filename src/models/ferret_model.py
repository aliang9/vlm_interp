import os
import logging
from typing import List, Optional, Tuple, Union, Dict, Any
import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel
from transformers import LlamaForCausalLM, LlamaConfig
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np

from ..utils.weight_manager import WEIGHTS_DIR, FERRET_DIR, PROJECTOR_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

class GeoRegionSampler(nn.Module):
    """Geometric region sampler to extract image patch features."""
    def __init__(self):
        super().__init__()
    
    def forward(self, images, boxes=None):
        """Extract features from image patches defined by boxes."""
        if boxes is None:
            return images
        
        patch_images = []
        for image, box in zip(images, boxes):
            x1, y1, x2, y2 = box
            x1 = max(0, min(x1, images.shape[3] - 1))
            y1 = max(0, min(y1, images.shape[2] - 1))
            x2 = max(0, min(x2, images.shape[3] - 1))
            y2 = max(0, min(y2, images.shape[2] - 1))
            
            patch = image[:, :, int(y1):int(y2), int(x1):int(x2)]
            
            patch = F.interpolate(patch, size=(images.shape[2], images.shape[3]), mode='bilinear')
            patch_images.append(patch)
        
        return torch.cat(patch_images, dim=0)

class FerretModel:
    """Ferret model for multimodal language understanding."""
    
    def __init__(self, model_path=None, device="cpu", use_4bit=False, use_7b=True):
        """Initialize the Ferret model.
        
        Args:
            model_path (str, optional): Path to the model weights. Defaults to None.
            device (str, optional): Device to run the model on. Defaults to "cpu".
            use_4bit (bool, optional): Whether to use 4-bit quantization. Defaults to False.
            use_7b (bool, optional): Whether to use 7B model. If False, use 13B. Defaults to True.
        """
        self.device = device
        self.use_4bit = use_4bit
        self.use_7b = use_7b
        
        self.model_path = model_path or str(FERRET_DIR)
        self.projector_path = str(PROJECTOR_DIR / "mm_projector.bin")
        
        logger.info(f"Initializing Ferret model from {self.model_path}")
        logger.info(f"Using projector from {self.projector_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_IMAGE_PATCH_TOKEN,
                DEFAULT_IM_START_TOKEN,
                DEFAULT_IM_END_TOKEN
            ]
        })
        
        self.vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
        self.vision_tower.to(self.device)
        
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        
        self.geo_sampler = GeoRegionSampler()
        
        if self.use_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_path, 
                torch_dtype=torch.float16,
                quantization_config=bnb_config
            )
        else:
            self.model = LlamaForCausalLM.from_pretrained(self.model_path)
        
        self.model.to(self.device)
        
        if os.path.exists(self.projector_path):
            logger.info(f"Loading projector from {self.projector_path}")
            projector_state_dict = torch.load(self.projector_path, map_location=self.device)
            
            self.mm_projector = nn.Linear(1024, self.model.config.hidden_size)
            self.mm_projector.load_state_dict(projector_state_dict)
            self.mm_projector.to(self.device)
        else:
            logger.warning(f"Projector not found at {self.projector_path}, creating new one")
            self.mm_projector = nn.Linear(1024, self.model.config.hidden_size)
            self.mm_projector.to(self.device)
    
    def process_image(self, image_path, region_coords=None):
        """Process an image for input to the model.
        
        Args:
            image_path (str): Path to the image file or URL.
            region_coords (List[float], optional): Region coordinates [x1, y1, x2, y2]. Defaults to None.
        
        Returns:
            torch.Tensor: Processed image features.
        """
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
        
        inputs = self.image_processor(image, return_tensors="pt").to(self.device)
        
        if region_coords is not None:
            region_coords = torch.tensor([region_coords])
            inputs = self.geo_sampler(inputs.pixel_values, region_coords)
        
        with torch.no_grad():
            image_features = self.vision_tower(**inputs).last_hidden_state
            image_features = self.mm_projector(image_features)
        
        return image_features
    
    def generate_response(self, prompt, image_path, max_new_tokens=512, temperature=0.7, region_coords=None):
        """Generate a response for the given prompt and image.
        
        Args:
            prompt (str): Text prompt.
            image_path (str): Path to the image file or URL.
            max_new_tokens (int, optional): Maximum number of new tokens. Defaults to 512.
            temperature (float, optional): Sampling temperature. Defaults to 0.7.
            region_coords (List[float], optional): Region coordinates [x1, y1, x2, y2]. Defaults to None.
        
        Returns:
            str: Generated response.
        """
        image_features = self.process_image(image_path, region_coords)
        
        if DEFAULT_IMAGE_TOKEN not in prompt:
            prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        image_token_pos = torch.where(input_ids == image_token_id)[1][0]
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                images=image_features,
                image_sizes=torch.tensor([[336, 336]]).to(self.device),
                image_token_positions=torch.tensor([image_token_pos]).to(self.device),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )
        
        response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        return response
