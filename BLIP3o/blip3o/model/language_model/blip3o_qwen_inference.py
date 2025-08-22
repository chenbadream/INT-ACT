from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3Model,
)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from blip3o.model.blip3o_arch import blip3oMetaForCausalLM, blip3oMetaModel
from blip3o.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
import numpy as np
from tqdm import tqdm
import PIL
from transformers import AutoTokenizer
from tok.mm_autoencoder import MMAutoEncoder

class blip3oQwenConfig(Qwen3Config):
    model_type = "blip3o_qwen_inference"

class blip3oQwenModel(blip3oMetaModel, Qwen3Model):
    config_class = blip3oQwenConfig

    def __init__(self, config: Qwen3Config):
        super(blip3oQwenModel, self).__init__(config)

class blip3oQwenForInferenceLM(Qwen3ForCausalLM, blip3oMetaForCausalLM):
    config_class = blip3oQwenConfig

    def __init__(self, config):
        Qwen3ForCausalLM.__init__(self, config)
        config.model_type = "blip3o_qwen"
        config.rope_scaling = None

        self.model = blip3oQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    @torch.no_grad()
    def prepare_inputs_labels_for_multimodal_inference(
        self,
        inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        image_sizes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simplified function for inference that only returns processed input_ids and attention_mask.
        Replaces all IMAGE_TOKEN_INDEX with actual image tokens.
        """
        if not isinstance(modalities, list):
            modalities = [modalities]
        
        pool_scale = 1

        # Encode images to get image tokens
        if images is not None:
            # Simplified for single image processing
            encoded_image_features = self.encode_images(images, modalities, pool_scale=pool_scale)
            image_tokens = encoded_image_features['image_tokens']
        else:
            image_tokens = None

        # Handle None inputs
        if attention_mask is None:
            attention_mask = torch.ones_like(inputs, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        # Remove padding using attention_mask
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(inputs, attention_mask)]

        # Since batch size is 1, we only process the first (and only) sequence
        cur_input_ids = input_ids[0]
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        
        # Find image token positions
        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        cur_input_ids_parts = []
        
        # Split input_ids around image tokens
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_parts.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
        
        # Reconstruct input_ids with image tokens
        cur_new_input_ids = []
        cur_new_attention_masks = []
        
        for i in range(num_images + 1):
            cur_new_input_ids.append(cur_input_ids_parts[i])
            # 对于原始文本部分，创建对应的attention_mask
            cur_new_attention_masks.append(torch.ones_like(cur_input_ids_parts[i], dtype=torch.bool))
            
            if i < num_images:
                if image_tokens is not None:
                    cur_image_tokens = image_tokens
                    
                    if pool_scale is not None:
                        pool_token = self.config.scale_start_token_id + pool_scale - 1
                        pool_token = torch.tensor([pool_token], dtype=torch.long, device=cur_image_tokens.device)
                        # 确保cur_image_tokens是一维的，以便与pool_token连接
                        cur_image_tokens = cur_image_tokens.flatten()
                        cur_image_tokens = torch.cat([pool_token, cur_image_tokens])
                    
                    cur_new_input_ids.append(cur_image_tokens)
                    # 为图像token创建对应的attention_mask
                    cur_new_attention_masks.append(torch.ones_like(cur_image_tokens, dtype=torch.bool))
        
        # Concatenate all parts
        new_input_ids = torch.cat(cur_new_input_ids)
        new_attention_mask = torch.cat(cur_new_attention_masks)
        
        # Truncate sequences to max length
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_ids = new_input_ids[:tokenizer_model_max_length]
            new_attention_mask = new_attention_mask[:tokenizer_model_max_length]
            
        # Ensure we return a tensor with batch dimension
        new_input_ids = new_input_ids.unsqueeze(0)
        new_attention_mask = new_attention_mask.unsqueeze(0)

        return new_input_ids, new_attention_mask

    @torch.no_grad()
    def generate_images_tokens(
        self,
        inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[torch.Tensor] = None,
        temperature: Optional[torch.Tensor] = None,
        top_p: Optional[torch.Tensor] = None,
        top_k: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        guidance_scale: float = 2.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        return_tensor=False,
        enable_progress_bar=False,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        
        # Replace all image tokens with actual image tokens
        inputs, attention_mask = self.prepare_inputs_labels_for_multimodal_inference(
                inputs, attention_mask, images, modalities, image_sizes=image_sizes
            )

        gen_ids = super(blip3oQwenForInferenceLM, self).generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            attention_mask=attention_mask,
            top_p=top_p,
            top_k=top_k)
        
        print(f"Generated token sequence shape: {gen_ids.shape}")

        # Find the start position of generated image tokens
        start_pos = (gen_ids == self.config.image_start_tag_id).float().argmax(dim=1)
        
        # Since batch size is 1, simplify the logic to directly process the single sequence
        b = 0
        start = start_pos[b].item() + 1  # Skip the start tag
        
        # Find end position if image_end_tag_id exists, otherwise use all remaining tokens
        if hasattr(self.config, 'image_end_tag_id'):
            end_positions = (gen_ids[b, start:] == self.config.image_end_tag_id).nonzero(as_tuple=True)[0]
            if len(end_positions) > 0:
                end = start + end_positions[0].item()
                batch_image_tokens = gen_ids[b, start:end]
            else:
                batch_image_tokens = gen_ids[b, start:]
        else:
            batch_image_tokens = gen_ids[b, start:]

        return batch_image_tokens
        

AutoConfig.register("blip3o_qwen_inference", blip3oQwenConfig)
AutoModelForCausalLM.register(blip3oQwenConfig, blip3oQwenForInferenceLM)
