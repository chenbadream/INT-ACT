from dataclasses import dataclass
import torch
from PIL import Image
from transformers import AutoTokenizer
from blip3o.model import *
import os
import numpy as np
from tok.mm_autoencoder import MMAutoEncoder
from huggingface_hub import hf_hub_download

@dataclass
class ImageTokenizerConfig:
    model_path: str = "/scratch/bc4227/INT-ACT-1/models/Overfit/checkpoint-6000"
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    # visual tokenizer config
    ar_path: str = 'ar_dtok_lp_256px.pth'
    encoder_path: str = 'ta_tok.pth'
    decoder_path: str = 'vq_ds16_t2i.pt'
    cfg_scale: float = 1.0

class ImageTokenizer:
    def __init__(self, config: ImageTokenizerConfig):
        self.config = config
        self.device = torch.device(config.device)
        self._load_models()
        
    def _load_models(self):
        self.model = blip3oQwenForInferenceLM.from_pretrained(self.config.model_path, torch_dtype=self.config.dtype).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        
        # Load visual tokenizer
        visual_config = dict(
            ar_path=self.config.ar_path,
            encoder_path=self.config.encoder_path,
            decoder_path=self.config.decoder_path,
            encoder_args={'input_type': 'rec'},
            decoder_args={},
        )
        self.visual_tokenizer = MMAutoEncoder(**visual_config).eval().to(dtype=self.config.dtype, device=self.device)

    def tokenize_image(self, image_path):
        """
        Tokenize an image using the vision tower
        """
        print(f"Tokenizing image: {image_path}")
        
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        processor = self.model.get_model().vision_tower.image_processor
        processed_image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        processed_image = processed_image.unsqueeze(0).to(self.device, dtype=self.config.dtype)
        
        # Use vision tower to get image tokens
        vision_tower = self.model.get_model().get_vision_tower()
        image_features = vision_tower(processed_image)
        
        # Extract tokens from the vision tower output
        image_tokens = image_features['tokens']

        print(f"Image tokens shape: {image_tokens.shape}")
        
        return {
            "image": image,
            "processed_image": processed_image,
            "image_features": image_features,
            "image_tokens": image_tokens
        }
    
    def detokenize_image(self, image_tokens, cfg_scale=1.0):
        """
        Detokenize image tokens back to an image using visual_tokenizer
        """
        print("Detokenizing image tokens...")
        reconstructed_image_tensor = self.visual_tokenizer.decode_from_encoder_indices(
            image_tokens, 
            {'cfg_scale': cfg_scale}
        )
        
        # Convert to PIL Image
        reconstructed_image = Image.fromarray(reconstructed_image_tensor[0].cpu().numpy())
        return reconstructed_image


def main():
    # Initialize configuration
    config = ImageTokenizerConfig()
    
    # Download necessary model weights
    config.ar_path = hf_hub_download("csuhan/TA-Tok", "ar_dtok_lp_1024px.pth")
    config.encoder_path = hf_hub_download("csuhan/TA-Tok", "ta_tok.pth")
    config.decoder_path = hf_hub_download("peizesun/llamagen_t2i", "vq_ds16_t2i.pt")
    
    # Initialize the image tokenizer
    image_tokenizer = ImageTokenizer(config)
    
    # Input image path
    image_path = "/vast/bc4227/datasets/bridge_processed/episode0000001/frame010.jpg"
    print(f"Input image: {image_path}")
    
    # Output directory
    output_dir = "BLIP3o-NEXT"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Tokenize the image using the vision tower
        print("\n=== Tokenizing Image with Vision Tower ===")
        result = image_tokenizer.tokenize_image(image_path)
        
        image = result["image"]
        image_tokens = result["image_tokens"]
        print(f"Image tokens shape: {image_tokens.shape}")
        
        # Detokenize the image back to pixel space
        print("\n=== Detokenizing Image Tokens ===")
        reconstructed_image = image_tokenizer.detokenize_image(image_tokens)
        
        # Save the original and reconstructed images
        original_save_path = os.path.join(output_dir, "original_image.png")
        image.save(original_save_path)
        print(f"Original image saved: {original_save_path}")
        
        reconstructed_save_path = os.path.join(output_dir, "reconstructed_image.png")
        reconstructed_image.save(reconstructed_save_path)
        print(f"Reconstructed image saved: {reconstructed_save_path}")
        
        print("\nImage tokenization and detokenization completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()  