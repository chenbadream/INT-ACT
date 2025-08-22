from dataclasses import dataclass
from blip3o.constants import IMAGE_TOKEN_INDEX
import torch
from PIL import Image
from transformers import AutoTokenizer
from blip3o.model import *
import os
import numpy as np
from tok.mm_autoencoder import MMAutoEncoder
from huggingface_hub import hf_hub_download
import re

@dataclass
class T2IConfig:
    model_path: str = "/scratch/bc4227/INT-ACT-1/models/Pretrain"
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    # generation config
    scale: int = 0  
    seq_len: int = 729  
    top_p: float = 0.95
    top_k: int = 1200
    # visual tokenizer config
    ar_path: str = 'ar_dtok_lp_256px.pth'
    encoder_path: str = 'ta_tok.pth'
    decoder_path: str = 'vq_ds16_t2i.pt'
    cfg_scale: float = 4.0

class TextToImageInference:
    def __init__(self, config: T2IConfig):
        self.config = config
        self.device = torch.device(config.device)
        self._load_models()
        
    def _load_models(self):
        self.model = blip3oQwenForInferenceLM.from_pretrained(self.config.model_path, torch_dtype=self.config.dtype).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        config = dict(
            ar_path=self.config.ar_path,
            encoder_path=self.config.encoder_path,
            decoder_path=self.config.decoder_path,
            encoder_args={'input_type': 'rec'},
            decoder_args={},
        )
        self.visual_tokenizer = MMAutoEncoder(**config).eval().to(dtype=self.config.dtype, device=self.device)
        self.visual_tokenizer.ar_model.cls_token_num = self.config.seq_len
        self.visual_tokenizer.encoder.pool_scale = self.config.scale + 1

    def process_image(self, image):
        """Process image using data_args image processor"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        processor = self.model.get_vision_tower().image_processor
        image_size = image.size
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size

    def generate_image(self, prompt: str, image_path: str = None) -> Image.Image:

        if 'image_token_index' not in globals():
            self.tokenizer.add_tokens(["<image>"], special_tokens=True)
            global image_token_index
            image_token_index = self.tokenizer.convert_tokens_to_ids("<image>")

        if image_path is None:
            raise ValueError("image_path is required for inference")
        
        print(f"Processing image: {image_path}")
        print(f"Instruction: {prompt}")

        batch_messages = []

        processed_image, image_size = self.process_image(image_path)
        images = processed_image.unsqueeze(0).to(self.device)
        image_sizes = [image_size]
        # 1 is for gen task (same as in dataset)
        modalities = [torch.tensor(1, device=self.device)] 


        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"<image>\nGiven this current view and the instruction '{prompt}', predict what the scene will look like in 10 steps."}
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)
        input_text += f"<im_start><S{self.config.scale}>"
        
        batch_messages.append(input_text)


        # tokenize as a batch
        inputs = self.tokenizer(batch_messages, return_tensors="pt", padding=True, truncation=True, padding_side="left")
        inputs.input_ids[inputs.input_ids == image_token_index] = IMAGE_TOKEN_INDEX

        gen_ids = self.model.generate_images_tokens(
            inputs.input_ids.to(self.device),
            inputs.attention_mask.to(self.device),
            images=images,
            image_sizes=image_sizes,
            modalities=modalities,
            max_new_tokens=self.config.seq_len,
            do_sample=True,
            top_p=self.config.top_p,
            top_k=self.config.top_k)

        gen_text = self.tokenizer.batch_decode(gen_ids)
        gen_code = [int(x[2:-1]) for x in gen_text if x.startswith("<I") and x.endswith(">")]
        gen_code = gen_code[:self.config.seq_len] + [0] * max(0, self.config.seq_len - len(gen_code))
        gen_code = torch.tensor(gen_code).unsqueeze(0).to(self.device)

        gen_tensor = self.visual_tokenizer.decode_from_encoder_indices(
            gen_code, 
            {'cfg_scale': self.config.cfg_scale}
        )
        gen_image = Image.fromarray(gen_tensor[0].numpy())
        return gen_image


def main():
    config = T2IConfig()
    config.ar_path = hf_hub_download("csuhan/TA-Tok", "ar_dtok_lp_1024px.pth")
    config.encoder_path = hf_hub_download("csuhan/TA-Tok", "ta_tok.pth")
    config.decoder_path = hf_hub_download("peizesun/llamagen_t2i", "vq_ds16_t2i.pt")
    inference = TextToImageInference(config)

    # Your specific instruction and image path
    instruction = "put the blue cube on the right side of the table on top of the rectangular block"
    image_path = "/vast/bc4227/datasets/bridge_processed/episode0000001/frame010.jpg"
    
    print(f"Instruction: {instruction}")
    print(f"Input image: {image_path}")
    
    output_dir = "BLIP3o-NEXT"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Generate the future image
        print("Generating future scene...")
        generated_image = inference.generate_image(instruction, image_path)
        
        if generated_image is not None:
            save_path = os.path.join(output_dir, "future_scene.png")
            generated_image.save(save_path)
            print(f"Successfully saved generated image: {save_path}")
            
            # Also save a copy with timestamp for reference
            import time
            timestamp = int(time.time())
            timestamped_path = os.path.join(output_dir, f"future_scene_{timestamp}.png")
            generated_image.save(timestamped_path)
            print(f"Timestamped copy saved: {timestamped_path}")
        else:
            print("Failed to generate image")
            
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    main()  