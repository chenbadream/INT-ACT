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
import torch.nn.functional as F
from torch import nn
from transformers import (
    AutoConfig,
    PaliGemmaForConditionalGeneration,
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

class PaliGemmaConfig(PretrainedConfig):
    model_type = "PaliGemmaModel"

    def __init__(
        self,
        paligemma_config: dict | None = None,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        attention_implementation: str = "eager",
        paligemma_pretrained_path: str | None = "google/paligemma-3b-pt-224",
        num_metaqueries: int = 64,
        **kwargs,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation
        self.num_metaqueries = num_metaqueries
        self.paligemma_pretrained_path = paligemma_pretrained_path

        if paligemma_config is None:
            # Default config from Pi0
            self.paligemma_config = CONFIG_MAPPING["paligemma"](
                transformers_version="4.48.1",
                _vocab_size=257152,
                bos_token_id=2,
                eos_token_id=1,
                hidden_size=2048,
                image_token_index=257152,
                model_type="paligemma",
                pad_token_id=0,
                projection_dim=2048,
                text_config={
                    "hidden_activation": "gelu_pytorch_tanh",
                    "hidden_size": 2048,
                    "intermediate_size": 16384,
                    "model_type": "gemma",
                    "num_attention_heads": 8,
                    "num_hidden_layers": 18,
                    "num_image_tokens": 256,
                    "num_key_value_heads": 1,
                    "torch_dtype": "float32",
                    "vocab_size": 257152,
                },
                vision_config={
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "model_type": "siglip_vision_model",
                    "num_attention_heads": 16,
                    "num_hidden_layers": 27,
                    "num_image_tokens": 256,
                    "patch_size": 14,
                    "projection_dim": 2048,
                    "projector_hidden_act": "gelu_fast",
                    "torch_dtype": "float32",
                    "vision_use_head": False,
                },
            )
        elif isinstance(self.paligemma_config, dict):
            # Override Pi0 default config for PaliGemma
            if "model_type" not in paligemma_config:
                paligemma_config["model_type"] = "paligemma"

            cfg_cls = CONFIG_MAPPING[paligemma_config["model_type"]]
            self.paligemma_config = cfg_cls(**paligemma_config)

        super().__init__(**kwargs)

    def __post_init__(self):
        super().__post_init__()
        if self.train_expert_only and not self.freeze_vision_encoder:
            raise ValueError(
                "You set `freeze_vision_encoder=False` and `train_expert_only=True` which are not compatible."
            )

        if self.attention_implementation not in ["eager", "fa2", "flex"]:
            raise ValueError(
                f"Wrong value provided for `attention_implementation` ({self.attention_implementation}). Expected 'eager', 'fa2' or 'flex'."
            )
        
class PaliGemmaModel(PreTrainedModel):
    config_class = PaliGemmaConfig

    def __init__(self, config: PaliGemmaConfig):
        super().__init__(config=config)
        self.config = config
        if config.paligemma_pretrained_path is not None:
            self.paligemma = PaliGemmaForConditionalGeneration.from_pretrained(config.paligemma_pretrained_path)
        else:
            self.paligemma = PaliGemmaForConditionalGeneration(config=config.paligemma_config)

        self.hidden_size = self.paligemma.config.text_config.hidden_size
        self.num_metaquery = config.num_metaqueries
        self.to_bfloat16_like_physical_intelligence()
        self.set_requires_grad()

    def set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
            for params in self.paligemma.vision_tower.parameters():
                params.requires_grad = False

        if self.config.train_expert_only:
            self.paligemma.eval()
            for params in self.paligemma.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.config.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()

        if self.config.train_expert_only:
            self.paligemma.eval()

    def to_bfloat16_like_physical_intelligence(self):
        self.paligemma = self.paligemma.to(dtype=torch.bfloat16)

        params_to_change_dtype = [
            "language_model.model.layers",
            "vision_tower",
            "multi_modal",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)

    def embed_image(self, image: torch.Tensor):
        # Handle different transformers versions
        if hasattr(self.paligemma, "get_image_features"):
            return self.paligemma.get_image_features(image)
        else:
            return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        embed_layer = self.paligemma.language_model.get_input_embeddings()
        return embed_layer(tokens)

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_emb: Optional[torch.FloatTensor] = None,
    ):
        models = self.paligemma.language_model.model
        num_metaquery = self.num_metaquery
        batch_size = inputs_emb.shape[0]
        metaquery_list = []
        prefix_q = []
        prefix_k = []
        prefix_v = []

        # RMSNorm
        num_layers = self.paligemma.config.text_config.num_hidden_layers # 18
        head_dim = self.paligemma.config.text_config.head_dim # 256
        for layer_idx in range(num_layers):
            layer = models.layers[layer_idx] 
            # normalizer = torch.tensor(models[i].config.hidden_size**0.5, dtype=hidden_states.dtype)
            # hidden_states = hidden_states * normalizer
            hidden_states = layer.input_layernorm(inputs_emb) #[B, (K * 256) + 48 + 2 + 64, 2048]

            input_shape = hidden_states.shape[:-1] # [B, (K * 256) + 48 + 2 + 64]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states = hidden_states.to(dtype=torch.bfloat16)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape) # [B, (K * 256) + 48 + 2 + 64, 8, 256]  
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape) # [B, (K * 256) + 48 + 2 + 64, 1, 256]  
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape) # [B, (K * 256) + 48 + 2 + 64, 1, 256]

            prefix_q.append(query_state[:, :-num_metaquery, :, :]) 
            prefix_k.append(key_state[:, :-num_metaquery, :, :])
            prefix_v.append(value_state[:, :-num_metaquery, :, :])

            query_state = apply_rope(query_state, position_ids) # [B, (K * 256) + 48 + 2 + 64, 8, 256]
            key_state = apply_rope(key_state, position_ids) # [B, (K * 256) + 48 + 2 + 64, 1, 256]

            attention_interface = self.get_attention_interface()
            att_output = attention_interface(
                attention_mask, batch_size, head_dim, query_state, key_state, value_state
            )
            att_output = att_output.to(dtype=torch.bfloat16) # [B, (K * 256) + 48 + 2 + 64, 1 * 8 * 256]

            if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
            out_emb = layer.self_attn.o_proj(att_output) # [B, (K * 256) + 48 + 2 + 64, 2048]

            out_emb += inputs_emb # [B, (K * 256) + 48 + 2 + 64, 2048]
            after_first_residual = out_emb.clone()

            out_emb = layer.post_attention_layernorm(out_emb) # [B, (K * 256) + 48 + 2 + 64, 2048]
            out_emb = layer.mlp(out_emb) # [B, (K * 256) + 48 + 2 + 64, 2048]

            # second residual
            out_emb += after_first_residual # [B, (K * 256) + 48 + 2 + 64, 2048]

            metaquery_list.append(out_emb[:, -num_metaquery:, :]) # [B, num_metaqueries, 2048]
            
            inputs_emb = out_emb

        # final norm
        out_emb = models.norm(out_emb) # [B, (K * 256) + 48 + 2 + 64, 2048]

        out_emb = out_emb[:, -self.num_metaquery:, :]  # [B, num_metaqueries, 2048]

        prompt_embeds = torch.cat(metaquery_list, dim=1)

        return prompt_embeds, prefix_q, prefix_k, prefix_v

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
        num_att_heads = self.config.paligemma_config.text_config.num_attention_heads # 8
        num_key_value_heads = self.config.paligemma_config.text_config.num_key_value_heads # 1
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
        
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)

        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        # probs: batch_size, num_key_value_head, num_att_head, sequence_length, sequence_length
        # value_states: batch_size, sequence_length, num_att_heads, head_dim

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)  # [B, (K * 256) + 48 + 1 + 50, 8, 256]
        # we use -1 because sequence length can change
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output # [B, (K * 256) + 48 + 1 + 50, 1 * 8 * 256] 
