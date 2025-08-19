from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import PIL
import numpy as np
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
from blip3o.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN
import numpy as np
import PIL


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

    def prepare_future_prediction_inputs(self, instruction: str, image_processor, tokenizer, current_image, future_step: int = 10):
        """
        Prepare inputs for future prediction following the same conversation template as the training data.
        
        Args:
            instruction: The task instruction
            image_processor: Image processor for the current image
            tokenizer: Tokenizer for text processing 
            current_image: PIL Image of current timestep
            future_step: Number of steps to predict into the future
            
        Returns:
            Dict with processed inputs ready for generation
        """
        # Create conversation following the training template
        conversation = [
            {
                "role": "user",
                "content": f"<image>\nGiven this current view and the instruction '{instruction}', predict what the scene will look like in {future_step} steps."
            },
            {
                "role": "assistant", 
                "content": f"Based on the instruction and current scene, the future view will be: {DEFAULT_IM_START_TOKEN}<image>{DEFAULT_IM_END_TOKEN}"
            }
        ]
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        
        # Process image
        if hasattr(image_processor, 'preprocess'):
            processed_image = image_processor.preprocess(current_image, return_tensors="pt")["pixel_values"]
        else:
            processed_image = image_processor(current_image, return_tensors="pt")["pixel_values"]
        
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "images": processed_image,
            "image_sizes": [current_image.size]
        }

    @torch.no_grad()
    def decode_image_tokens(self, image_tokens, normalize=True, return_tensor=False):
        """
        Decode discrete image tokens back to images using the vision tower's decoder.
        
        Args:
            image_tokens: tensor of shape (batch_size, num_tokens) containing discrete image token indices
            normalize: whether to normalize the output to [0, 1] range
            return_tensor: whether to return tensor or PIL images
        
        Returns:
            PIL images or tensor depending on return_tensor flag
        """
        vision_tower = self.get_vision_tower()
        if vision_tower is None:
            raise ValueError("Vision tower is not available for image decoding")
        
        # Convert token indices back to the original codebook indices
        # During encoding: image_tokens = tokens + self.config.image_start_token_id
        # So to decode: tokens = image_tokens - self.config.image_start_token_id
        original_indices = image_tokens - self.config.image_start_token_id
        
        # Use the TextAlignedTokenizer's decode_from_bottleneck method
        # This handles the full decode pipeline: indices -> regularizer.decode -> bottleneck.decode -> spatial decode
        try:
            decoded_features = vision_tower.vision_tower.decode_from_bottleneck(original_indices)
            
            # The decoded_features are in the encoder's hidden dimension space
            # We need to convert them back to pixel space
            # Apply the scaling layer inverse to get back to pixel values
            decoded_images = vision_tower.vision_tower.scale_layer.inv(decoded_features)
            
        except Exception as e:
            print(f"Error during decode_from_bottleneck: {e}")
            # Fallback: manual decoding process
            try:
                # Step 1: Decode through the regularizer (bottleneck indices -> quantized features)
                regularizer = vision_tower.vision_tower.bottleneck.regularizer
                quantized_features = regularizer.decode(original_indices)  # (batch_size, num_tokens, bottleneck_dim)
                
                # Step 2: Project back through bottleneck
                decoder_input = vision_tower.vision_tower.bottleneck.project_out(quantized_features)
                
                # Step 3: Spatial decoding through the decoder
                batch_size = decoder_input.shape[0]
                num_tokens = decoder_input.shape[1]
                spatial_size = int(num_tokens ** 0.5)
                
                # Reshape to spatial format for the decoder
                decoder_input = decoder_input.view(batch_size, spatial_size, spatial_size, -1)
                decoder_input = decoder_input.permute(0, 3, 1, 2)  # (batch, channels, height, width)
                
                # Use the decoder
                decoded_features = vision_tower.vision_tower.decode(decoder_input)
                
                # Apply inverse scaling
                decoded_images = vision_tower.vision_tower.scale_layer.inv(decoded_features)
                
            except Exception as e2:
                print(f"Fallback decoding also failed: {e2}")
                raise ValueError(f"Could not decode image tokens: {e}, {e2}")
        
        # Normalize to [0, 1] range if requested
        if normalize:
            decoded_images = torch.clamp(decoded_images, 0, 1)
        
        if return_tensor:
            return decoded_images
        
        # Convert to PIL images
        decoded_images = decoded_images.cpu().permute(0, 2, 3, 1).float().numpy()
        pil_images = []
        for img in decoded_images:
            if normalize:
                img_uint8 = (img * 255).round().astype("uint8")
            else:
                img_uint8 = np.clip(img * 255, 0, 255).round().astype("uint8")
            pil_images.append(PIL.Image.fromarray(img_uint8))
        
        return pil_images



    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    @torch.no_grad()
    def generate_future_images(
        self,
        inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 0.9,
        top_k: Optional[int] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        return_tensor: bool = False,
        **kwargs,
    ):
        """
        Generate future images by first generating discrete image tokens, then decoding them to images.
        
        This method follows the future prediction conversation template where:
        1. Current image and instruction are given as input
        2. Model generates future view with image tokens between <im_start> and <im_end>
        
        Args:
            inputs: input token ids
            attention_mask: attention mask for inputs
            max_new_tokens: maximum number of new tokens to generate
            temperature: sampling temperature
            top_p: nucleus sampling parameter
            top_k: top-k sampling parameter
            images: input images (current timestep)
            image_sizes: sizes of input images
            modalities: modalities for input
            return_tensor: whether to return images as tensors or PIL images
            **kwargs: additional generation parameters
            
        Returns:
            tuple of (generated_ids, decoded_images)
        """
        position_ids = kwargs.pop("position_ids", None)
        
        # Prepare inputs with multimodal processing if images are provided
        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        
        # Generate text tokens including discrete image tokens
        generated_ids = super(blip3oQwenForInferenceLM, self).generate(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **kwargs
        )
        
        # Extract image tokens from the generated sequence
        # Look for the pattern: <im_start><image_tokens><im_end>
        start_token_id = getattr(self.config, 'image_start_tag_id', None)  # <im_start> token
        end_token_id = getattr(self.config, 'image_end_tag_id', None)      # <im_end> token
        
        if start_token_id is None or end_token_id is None:
            print("Warning: Image start/end token IDs not found in config")
            # Try to find from constants or tokenizer
            try:
                from blip3o.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
                tokenizer = getattr(self, 'tokenizer', None)
                if tokenizer is not None:
                    start_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
                    end_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
                else:
                    print("No tokenizer available to find image token IDs")
                    return generated_ids, []
            except ImportError:
                print("Could not import constants for image tokens")
                return generated_ids, []
        
        # Find positions of image start and end tokens
        start_positions = (generated_ids == start_token_id).nonzero(as_tuple=True)
        end_positions = (generated_ids == end_token_id).nonzero(as_tuple=True)
        
        if len(start_positions[0]) == 0 or len(end_positions[0]) == 0:
            print("No image tokens generated in the sequence")
            return generated_ids, []
        
        # Extract image tokens for each sequence in the batch
        decoded_images = []
        for batch_idx in range(generated_ids.size(0)):
            # Find start and end positions for this batch
            batch_start_positions = start_positions[1][start_positions[0] == batch_idx]
            batch_end_positions = end_positions[1][end_positions[0] == batch_idx]
            
            if len(batch_start_positions) == 0 or len(batch_end_positions) == 0:
                print(f"No matching image tokens for batch {batch_idx}")
                continue
                
            # Get the last generated image (most recent future prediction)
            # In the conversation template, the assistant responds with: "Based on... <im_start><image><im_end>"
            start_pos = batch_start_positions[-1].item() + 1  # +1 to skip the start token
            end_pos = batch_end_positions[-1].item()  # end token is exclusive
            
            if start_pos >= end_pos:
                print(f"Invalid image token positions for batch {batch_idx}: start={start_pos}, end={end_pos}")
                continue
                
            # Extract image tokens for this sequence
            image_tokens = generated_ids[batch_idx, start_pos:end_pos]
            
            print(f"Extracted {len(image_tokens)} image tokens for batch {batch_idx}")
            
            # Decode image tokens to image
            try:
                decoded_image = self.decode_image_tokens(
                    image_tokens.unsqueeze(0),  # Add batch dimension
                    normalize=True,
                    return_tensor=return_tensor
                )
                if return_tensor:
                    decoded_images.append(decoded_image[0])  # Remove batch dimension
                else:
                    decoded_images.extend(decoded_image)  # PIL images
                    
                print(f"Successfully decoded image for batch {batch_idx}")
                    
            except Exception as e:
                print(f"Warning: Failed to decode image tokens for batch {batch_idx}: {e}")
                continue
        
        return generated_ids, decoded_images

    def predict_future_image(
        self,
        instruction: str,
        current_image,  # PIL Image or tensor
        tokenizer,
        future_step: int = 10,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        return_tensor: bool = False,
    ):
        """
        End-to-end method to predict future images given current image and instruction.
        
        Args:
            instruction: Task instruction string
            current_image: PIL Image of current timestep
            tokenizer: Tokenizer to use
            future_step: Number of steps to predict ahead
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            return_tensor: Whether to return tensor or PIL images
            
        Returns:
            List of predicted future images (PIL Images or tensors)
        """
        import PIL.Image
        
        # Convert tensor to PIL if needed
        if not isinstance(current_image, PIL.Image.Image):
            if torch.is_tensor(current_image):
                # Assume tensor is in (C, H, W) format and in [0, 1] range
                current_image = current_image.cpu().permute(1, 2, 0).numpy()
                current_image = (current_image * 255).astype(np.uint8)
                current_image = PIL.Image.fromarray(current_image)
            else:
                raise ValueError("current_image must be PIL Image or torch tensor")
        
        # Get vision tower's image processor
        vision_tower = self.get_vision_tower()
        if vision_tower is None:
            raise ValueError("Vision tower not available")
        
        image_processor = vision_tower.image_processor
        
        # Prepare inputs using the conversation template
        inputs_dict = self.prepare_future_prediction_inputs(
            instruction, image_processor, tokenizer, current_image, future_step
        )
        
        # Move to device
        device = next(self.parameters()).device
        inputs_dict = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs_dict.items()}
        
        # Generate future images
        generated_ids, decoded_images = self.generate_future_images(
            inputs=inputs_dict["input_ids"],
            attention_mask=inputs_dict["attention_mask"],
            images=inputs_dict["images"],
            image_sizes=inputs_dict["image_sizes"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            return_tensor=return_tensor,
        )
        
        return decoded_images








AutoConfig.register("blip3o_qwen_inference", blip3oQwenConfig)
AutoModelForCausalLM.register(blip3oQwenConfig, blip3oQwenForInferenceLM)

