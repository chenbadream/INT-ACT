# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Union

import torch
import torch.version
from pytest import Cache
from torch import nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    GemmaForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto import CONFIG_MAPPING

from src.model.mmmvla.flex_attention import flex_attention_forward

def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)

class ActionExpertConfig(PretrainedConfig):
    model_type = "action_expert"

    def __init__(
        self,
        gemma_expert_config: dict | None = None,
        attention_implementation: str = "eager",
        **kwargs,
    ):
        self.attention_implementation = attention_implementation

        if gemma_expert_config is None:
            # Default config from Pi0
            self.gemma_expert_config = CONFIG_MAPPING["gemma"](
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=256,
                hidden_act="gelu_pytorch_tanh",
                hidden_activation="gelu_pytorch_tanh",
                hidden_size=1024,
                initializer_range=0.02,
                intermediate_size=4096,
                max_position_embeddings=8192,
                model_type="gemma",
                num_attention_heads=8,
                num_hidden_layers=18,
                num_key_value_heads=1,
                pad_token_id=0,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype="float32",
                transformers_version="4.48.1",
                use_cache=True,
                vocab_size=257153,
            )
        elif isinstance(self.gemma_expert_config, dict):
            # Override Pi0 default config for Gemma Expert
            if "model_type" not in gemma_expert_config:
                gemma_expert_config["model_type"] = "gemma"

            cfg_cls = CONFIG_MAPPING[gemma_expert_config["model_type"]]
            self.gemma_expert_config = cfg_cls(**gemma_expert_config)

        super().__init__(**kwargs)

class ActionExpertModel(PreTrainedModel):
    config_class = ActionExpertConfig

    def __init__(self, config: ActionExpertConfig):
        super().__init__(config=config)
        self.config = config

        # Initialize the Gemma Expert model
        self.gemma_expert = GemmaForCausalLM(config=config.gemma_expert_config)
        # Remove unused embed_tokens
        self.gemma_expert.model.embed_tokens = None
        self.hidden_size = self.gemma_expert.config.hidden_size # 1024
        self.head_dim = self.gemma_expert.config.head_dim # 256
        self.to_bfloat16_like_physical_intelligence()

    def to_bfloat16_like_physical_intelligence(self):

        params_to_change_dtype = [
            "gemma_expert.model.layers",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)
    
    def forward_attn_layer(
        self,
        model_layers,
        layer_idx,
        suffix_embeds,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
    ) -> torch.Tensor:
        """
        Self-attention layer for suffix embeddings only.
        Args:
            suffix_embeds: Input embeddings for the suffix sequence [B, L_suffix, D]
            layer_idx: Current layer index
            position_ids: Position IDs for the suffix sequence [B, L_suffix]
            attention_mask: Attention mask for suffix sequence [B, L_suffix, L_suffix]
            batch_size: Batch size
            head_dim: Head dimension
        Returns:
            Attention output for suffix embeddings
        """
        # Get the expert layer
        layer = model_layers[layer_idx] 
        # Apply layer normalization
        hidden_states = layer.input_layernorm(suffix_embeds)
        
        # Compute Q, K, V for suffix embeddings
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        
        hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
        query_states = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
        key_states = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
        value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape)
        
        # Apply RoPE to Q and K
        query_states = apply_rope(query_states, position_ids)
        key_states = apply_rope(key_states, position_ids)
        
        # Perform self-attention
        attention_interface = self.get_attention_interface()
        att_output = attention_interface(
            attention_mask, batch_size, head_dim, query_states, key_states, value_states
        )
        
        # Apply output projection
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        att_output = layer.self_attn.o_proj(att_output)
        
        return att_output
    
    def forward_cross_attn_layer(
        self,
        model_layers,
        suffix_embeds,
        prompt_embeds,
        layer_idx,
        suffix_position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values=None,
    ) -> torch.Tensor:
        """
        Cross-attention layer where suffix embeddings attend to prompt embeddings.
        If suffix_embeds is None, only pre-computes and caches K,V from prompt_embeds.
        """
        attention_interface = self.get_attention_interface()
        
        # Get the expert layer
        layer = model_layers[layer_idx] 
        
        # Handle KV cache for prompt embeddings
        if use_cache and past_key_values is None:
            past_key_values = {}
            
        if use_cache:
            if fill_kv_cache:
                # Pre-compute and cache K, V from prompt embeddings
                prompt_input_shape = prompt_embeds.shape[:-1]
                prompt_hidden_shape = (*prompt_input_shape, -1, layer.self_attn.head_dim)
                
                prompt_embeds_projected = prompt_embeds.to(dtype=layer.self_attn.k_proj.weight.dtype)
                
                # Check dimension compatibility
                if prompt_embeds_projected.shape[-1] != layer.self_attn.k_proj.in_features:
                    raise ValueError(
                        f"Prompt embeddings dimension {prompt_embeds_projected.shape[-1]} "
                        f"doesn't match K projection input {layer.self_attn.k_proj.in_features}"
                    )
                
                key_states = layer.self_attn.k_proj(prompt_embeds_projected).view(prompt_hidden_shape)
                value_states = layer.self_attn.v_proj(prompt_embeds_projected).view(prompt_hidden_shape)
                
                if layer_idx not in past_key_values:
                    past_key_values[layer_idx] = {}
                past_key_values[layer_idx]["key_states"] = key_states
                past_key_values[layer_idx]["value_states"] = value_states
            else:
                # Use cached K, V from prompt embeddings
                key_states = past_key_values[layer_idx]["key_states"]
                value_states = past_key_values[layer_idx]["value_states"]
        else:
            # No cache, compute K, V from prompt embeddings each time
            prompt_input_shape = prompt_embeds.shape[:-1]
            prompt_hidden_shape = (*prompt_input_shape, -1, layer.self_attn.head_dim)
            
            prompt_embeds_projected = prompt_embeds.to(dtype=layer.self_attn.k_proj.weight.dtype)
            key_states = layer.self_attn.k_proj(prompt_embeds_projected).view(prompt_hidden_shape)
            value_states = layer.self_attn.v_proj(prompt_embeds_projected).view(prompt_hidden_shape)
        
        # If suffix_embeds is None, only return cached K,V without computing attention
        if suffix_embeds is None:
            return None, past_key_values
        
        # Process suffix embeddings for Query
        suffix_hidden_states = layer.input_layernorm(suffix_embeds)
        suffix_input_shape = suffix_hidden_states.shape[:-1]
        suffix_hidden_shape = (*suffix_input_shape, -1, layer.self_attn.head_dim)
        
        suffix_hidden_states = suffix_hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
        query_states = layer.self_attn.q_proj(suffix_hidden_states).view(suffix_hidden_shape)
        
        # Apply RoPE to queries using original suffix_position_ids
        query_states = apply_rope(query_states, suffix_position_ids)
        
        # Perform cross-attention: Q from suffix, K,V from prompt
        att_output = attention_interface(
            attention_mask, batch_size, head_dim, query_states, key_states, value_states
        )
        
        # Apply output projection
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        att_output = layer.self_attn.o_proj(att_output)
        
        return att_output, past_key_values

    def forward(
        self,
        suffix_embeds: Optional[torch.FloatTensor] = None,
        prompt_embeds: torch.FloatTensor = None,
        suffix_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        suffix_position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
        alternate_pattern: str = "self_cross",  # "self_cross" or "cross_self"
    ):
        """
        Forward pass with alternating self-attention and cross-attention using only expert model.
        Self-attention layers do not use KV cache for simplicity.
        
        Args:
            suffix_embeds: Input embeddings for suffix sequence [B, L_suffix, D]. 
                         If None, only computes and caches K,V from prompt_embeds for cross-attention layers.
            prompt_embeds: Pre-computed prompt embeddings [B, L_prompt, D] 
            suffix_attention_mask: Self-attention mask for suffix [B, L_suffix, L_suffix]
            cross_attention_mask: Cross-attention mask [B, L_suffix, L_prompt]
            suffix_position_ids: Position IDs for suffix [B, L_suffix]
            past_key_values: Cached key-value pairs (only for cross-attention)
            use_cache: Whether to use KV cache for cross-attention
            fill_kv_cache: Whether to fill the KV cache for cross-attention
            alternate_pattern: Pattern for alternating attention ("self_cross" or "cross_self")
        
        Returns:
            Output embeddings and updated past_key_values. If suffix_embeds is None,
            returns (None, past_key_values) with only the cached K,V values.
        """
        # If suffix_embeds is None, only pre-compute and cache K,V from prompt_embeds
        if suffix_embeds is None:
            if prompt_embeds is None:
                raise ValueError("Either suffix_embeds or prompt_embeds must be provided")
            
            # Pre-compute K,V cache for all cross-attention layers
            model = self.gemma_expert.model
            num_layers = len(model.layers)
            past_key_values = {}
            
            for layer_idx in range(num_layers):
                # Determine if this layer is a cross-attention layer
                if alternate_pattern == "self_cross":
                    is_cross_attention = (layer_idx % 2 == 1)  # cross at odd indices
                elif alternate_pattern == "cross_self":
                    is_cross_attention = (layer_idx % 2 == 0)  # cross at even indices
                else:
                    raise ValueError(f"Unknown alternate_pattern: {alternate_pattern}")
                
                if is_cross_attention:
                    # Pre-compute K, V for this cross-attention layer
                    _, past_key_values = self.forward_cross_attn_layer(
                        model.layers,
                        None,  # suffix_embeds is None
                        prompt_embeds,
                        layer_idx,
                        None,  # suffix_position_ids not needed
                        None,  # attention_mask not needed
                        prompt_embeds.shape[0],  # batch_size
                        self.head_dim,
                        use_cache=True,
                        fill_kv_cache=True,
                        past_key_values=past_key_values,
                    )
            
            return None, past_key_values
        
        model = self.gemma_expert.model
        batch_size = suffix_embeds.shape[0]
        head_dim = self.head_dim

        # Initialize current embeddings
        current_embeds = suffix_embeds
        
        # Process through all layers with alternating pattern
        num_layers = len(model.layers) 

        for layer_idx in range(num_layers):
            # Determine attention type based on alternating pattern
            if alternate_pattern == "self_cross":
                # Pattern: self, cross, self, cross, ...
                use_self_attention = (layer_idx % 2 == 0)
            elif alternate_pattern == "cross_self":
                # Pattern: cross, self, cross, self, ...
                use_self_attention = (layer_idx % 2 == 1)
            else:
                raise ValueError(f"Unknown alternate_pattern: {alternate_pattern}")
            
            if use_self_attention:
                # Self-attention layer (always no cache)
                att_output = self.forward_attn_layer(
                    model.layers,
                    layer_idx,
                    current_embeds,
                    suffix_position_ids,
                    suffix_attention_mask,
                    batch_size,
                    head_dim,
                )
            else:
                # Cross-attention layer (with cache for prompt K,V)
                att_output, past_key_values = self.forward_cross_attn_layer(
                    model.layers,
                    current_embeds,
                    prompt_embeds,
                    layer_idx,
                    suffix_position_ids,
                    cross_attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            
            # Post-processing: residual connection + MLP
            layer = model.layers[layer_idx]

            # First residual connection
            out_emb = att_output + current_embeds
            
            # Store for second residual connection
            residual = out_emb
            
            # MLP block
            out_emb = layer.post_attention_layernorm(out_emb)
            out_emb = layer.mlp(out_emb)
            
            # Second residual connection
            out_emb = out_emb + residual
            
            # Update current embeddings for next layer
            current_embeds = out_emb
        
        # Final layer normalization
        final_embeds = model.norm(current_embeds)
        
        return final_embeds, past_key_values

    def get_attention_interface(self):
        if self.config.attention_implementation == "fa2":
            attention_interface = self.flash_attention_forward
        elif self.config.attention_implementation == "flex":
            attention_interface = flex_attention_forward
        else:
            attention_interface = self.eager_attention_forward
        return attention_interface

    def flash_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        raise NotImplementedError("FA2 is not implemented (yet)")

    def eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        num_att_heads = self.config.gemma_expert_config.num_attention_heads # 8
        num_key_value_heads = self.config.gemma_expert_config.num_key_value_heads # 1
        num_key_value_groups = num_att_heads // num_key_value_heads # 8 // 1 = 8

        # query_states: batch_size, sequence_length, num_att_head, head_dim
        # key_states: batch_size, sequence_length, num_key_value_head, head_dim
        # value_states: batch_size, sequence_length, num_key_value_head, head_dim
        sequence_length = key_states.shape[1] # (K * 256) + 48 + 1 + 50

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        ) # [B, (K * 256) + 48 + 1 + 50, 8, 256]

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        ) # [B, (K * 256) + 48 + 1 + 50, 8, 256]

        # Attention here is upcasted to float32 to match the original eager implementation.

        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2) # [B, 8, (K * 256) + 48 + 1 + 50, 256]
        key_states = key_states.transpose(1, 2) # [B, 8, (K * 256) + 48 + 1 + 50, 256]

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3)) 
        att_weights *= head_dim**-0.5
        big_neg = -2.3819763e38  # See gemma/modules.py

        att_weights = att_weights.to(device=attention_mask.device)
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)

        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype, device=value_states.device) # [B, 1, 8, (K * 256) + 48 + 1 + 50, (K * 256) + 48 + 1 + 50]

        # probs: batch_size, num_key_value_head, num_att_head, sequence_length, sequence_length
        # value_states: batch_size, sequence_length, num_att_heads, head_dim

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)  # [B, (K * 256) + 48 + 1 + 50, 8, 256]
        # we use -1 because sequence length can change
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output # [B, (K * 256) + 48 + 1 + 50, 1 * 8 * 256]