#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""
π0: A Vision-Language-Action Flow Model for General Robot Control

[Paper](https://www.physicalintelligence.company/download/pi0.pdf)
[Jax code](https://github.com/Physical-Intelligence/openpi)

Designed by Physical Intelligence. Ported from Jax by Hugging Face.

Install pi0 extra dependencies:
```bash
pip install -e ".[pi0]"
```

Example of finetuning the pi0 pretrained model (`pi0_base` in `openpi`):
```bash
python -m lerobot.scripts.train \
--policy.path=lerobot/pi0 \
--dataset.repo_id=danaaubakirova/koch_test
```

Example of finetuning the pi0 neural network with PaliGemma and expert Gemma
pretrained with VLM default parameters before pi0 finetuning:
```bash
python -m lerobot.scripts.train \
--policy.type=pi0 \
--dataset.repo_id=danaaubakirova/koch_test
```

Example of using the pi0 pretrained model outside LeRobot training framework:
```python
policy = Pi0Policy.from_pretrained("lerobot/pi0")
```

"""

import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoTokenizer, Qwen2Config
from numpy import true_divide

from src.model.mvla.transformer_encoder import Qwen2Encoder
from lerobot.common.constants import ACTION, OBS_ROBOT
from lerobot.common.policies.normalize import Normalize, Unnormalize
from src.model.mvla.configuration_mvla import MVLAConfig
from src.model.mvla.mllm import (PaliGemmaModel, PaliGemmaConfig)
from src.model.mmmvla.expert import (ActionExpertConfig, ActionExpertModel)
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.utils import get_safe_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class MVLAPolicy(PreTrainedPolicy):
    """Wrapper class around PI0FlowMatching model to train and run inference within LeRobot."""

    config_class = MVLAConfig
    name = "mvla"

    def __init__(
        self,
        config: MVLAConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.model = PI0FlowMatching(config)
        self.language_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

        self.reset()


    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    @torch.no_grad
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError("Currently not implemented for MVLA")

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        if self.config.adapt_to_pi_aloha:
            batch[OBS_ROBOT] = self._pi_aloha_decode_state(batch[OBS_ROBOT])

        batch = self.normalize_inputs(batch)

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            images, img_masks = self.prepare_images(batch)
            state = self.prepare_state(batch)
            lang_tokens, lang_masks = self.prepare_language(batch)

            actions = self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=noise
            ) # (batch_size, 50, 32)

            # Unpad actions
            original_action_dim = self.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]

            actions = self.unnormalize_outputs({"action": actions})["action"]

            if self.config.adapt_to_pi_aloha:
                actions = self._pi_aloha_encode_actions(actions)

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict[str, Tensor]]:
        """Do a full training forward pass to compute the loss"""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_ROBOT] = self._pi_aloha_decode_state(batch[OBS_ROBOT])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        loss_dict = {}
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time) # (B, 50, 32)
        loss_dict["losses_after_forward"] = losses.clone()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone()

        # For backward pass
        loss = losses.mean()
        # For logging
        loss_dict["l2_loss"] = loss.detach() 

        return loss, loss_dict

    def prepare_images(self, batch):
        """Apply Pi0 preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks  # list of (batch_size, 3, 224, 224) tensors, list of (batch_size,) tensors

    @torch.compiler.disable(recursive=False)
    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Tokenize the text input"""
        device = batch[OBS_ROBOT].device
        tasks = batch["task"]
        # print("tasks:", tasks)
        # PaliGemma prompt has to end with a new line
        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding="max_length",
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
            truncation=True, # Irving: add truncation to follow Allen
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = pad_vector(batch[OBS_ROBOT], self.config.max_state_dim)
        return state # (batch_size, 32)

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions # (batch_size, n_action_steps, 32)


class PI0FlowMatching(nn.Module):
    """
    π0: A Vision-Language-Action Flow Model for General Robot Control

    [Paper](https://www.physicalintelligence.company/download/pi0.pdf)
    [Jax code](https://github.com/Physical-Intelligence/openpi)

    Designed by Physical Intelligence. Ported from Jax by Hugging Face.
    ┌──────────────────────────────┐
    │               actions        │
    │               ▲              │
    │              ┌┴─────┐        │
    │  kv cache    │Gemma │        │
    │  ┌──────────►│Expert│        │
    │  │           │      │        │
    │ ┌┴────────┐  │x 10  │        │
    │ │         │  └▲──▲──┘        │
    │ │PaliGemma│   │  │           │
    │ │         │   │  robot state │
    │ │         │   noise          │
    │ └▲──▲─────┘                  │
    │  │  │                        │
    │  │  image(s)                 │
    │  language tokens             │
    └──────────────────────────────┘
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        paligemmaconfig = PaliGemmaConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
            paligemma_pretrained_path=self.config.paligemma_pretrained_path,
            num_metaqueries=self.config.num_metaqueries,
        )
    
        self.paligemma = PaliGemmaModel(paligemmaconfig)
        actionexpertconfig = ActionExpertConfig()
        self.action_expert = ActionExpertModel(actionexpertconfig)

        self.metaquery = nn.Parameter(torch.zeros(1, self.config.num_metaqueries, 2048, dtype=torch.bfloat16))
        torch.nn.init.normal_(self.metaquery, std=.02)

        # self.connector = nn.Linear(self.paligemma.hidden_size, self.action_expert.hidden_size)
        """
        # Initialize the Action Model (DiT-based diffusion model) directly
        action_model_config = {
            "token_size": self.config.action_model_token_size,  # Match the number of metaqueries
            "model_type": self.config.action_model_type,
            "in_channels": self.config.max_action_dim,  # Use the actual action dimension
            "future_action_window_size": self.config.n_action_pred_tokens - 1,  # Use the number of action steps
            "past_action_window_size": 0,  # Must be 0 as action history is not used
            "diffusion_steps": self.config.action_model_diffusion_steps,
            "noise_schedule": self.config.action_model_noise_schedule
        }
        self.action_model = ActionModel(**action_model_config)
        
        # Initialize action prediction tokens
        self.action_pred_token = nn.Parameter(torch.zeros(1, self.config.n_action_pred_tokens, self.paligemma.hidden_size))
        torch.nn.init.normal_(self.action_pred_token, std=.02)
        """
        self.connector_in_dim = self.paligemma.hidden_size
        self.connector_out_dim = self.action_expert.hidden_size
        self.connector_num_hidden_layers = self.config.connector_num_hidden_layers
        
        encoder = Qwen2Encoder(
            Qwen2Config(
                hidden_size=self.connector_in_dim,
                intermediate_size=self.connector_in_dim * 4,
                num_hidden_layers=self.connector_num_hidden_layers,  # 12
                num_attention_heads=self.connector_in_dim // 64,
                num_key_value_heads=self.connector_in_dim // 64,
                initializer_range=0.014,
                use_cache=False,
                rope=True,
                qk_norm=True,
            ),
        )
        self.connector = nn.Sequential(
            encoder,
            nn.Linear(self.connector_in_dim, self.connector_out_dim),
            nn.LayerNorm(self.connector_out_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.connector_out_dim, self.connector_out_dim),
            nn.LayerNorm(self.connector_out_dim),
        )
        
        # Projections are not needed for DiT head
        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)   #(32, 1024)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)  #(32, 1024)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)  #(1024, 32)

        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)  #(2048, 1024)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)  #(1024, 1024)

        self.set_requires_grad()

        
        # Gradient checkpointing
        if config._gradient_checkpointing:
            try:
                self.paligemma.gradient_checkpointing_enable(
                    {"use_reentrant": False}
                )
            except:
                pass
            if not isinstance(self.connector, nn.Identity):
                for module in self.connector:
                    if isinstance(module, Qwen2Encoder):
                        module.gradient_checkpointing_enable({"use_reentrant": False})
            self.paligemma._set_gradient_checkpointing()
        

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)  

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        for (
            img,
            img_mask,
        ) in zip(images, img_masks, strict=False):
            img_emb = self.paligemma.embed_image(img)
            img_emb = img_emb.to(dtype=torch.bfloat16)

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        lang_emb = self.paligemma.embed_language_tokens(lang_tokens)

        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        # Add action prediction tokens
        action_pred_emb = self.metaquery.expand(bsize, -1, -1)
        action_pred_emb = action_pred_emb.to(dtype=torch.bfloat16)
        embs.append(action_pred_emb)

        action_pred_mask = torch.ones(bsize, self.config.num_metaqueries, dtype=torch.bool, device=lang_masks.device)
        pad_masks.append(action_pred_mask)
        
        # Set attention masks so that action prediction tokens can attend to all previous tokens
        att_masks += [1] + [0] * (self.config.num_metaqueries - 1)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed state
        state_emb = self.state_proj(state)
        state_emb = state_emb.to(dtype=torch.bfloat16)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        dtype = state_emb.dtype
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks # [B, 1 + 50, 1024], [B, 1 + 50], [B, 1 + 50] 

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device) # (batch_size, 50, 32)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device) # (batch_size, )

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, time)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks) # [B, (K * 256) + 48 + 2 + 64, (K * 256) + 48 + 2 + 64]
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1 # [B, (K * 256) + 48 + 2 + 64]

        prompt_embs = self.paligemma.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            inputs_emb=prefix_embs,
        )

        prompt_embs = self.connector(prompt_embs)
        suffix_len = suffix_pad_masks.shape[1] # 1 + 50
        batch_size = prompt_embs.shape[0] # B
        prefix_len = prompt_embs.shape[1] # 64

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks) # [B, 1 + 50, 1 + 50]
        suffix_position_ids = torch.cumsum(suffix_pad_masks, dim=1) - 1 # [B, 1 + 50]

        cross_attention_mask = torch.ones((batch_size, suffix_len, prefix_len), dtype=torch.bool, device=suffix_att_2d_masks.device)

        suffix_out, _ = self.action_expert.forward(
            suffix_embeds=suffix_embs,
            prompt_embeds=prompt_embs,
            suffix_attention_mask=suffix_att_2d_masks,
            cross_attention_mask=cross_attention_mask,
            suffix_position_ids=suffix_position_ids,
            past_key_values=None,
            use_cache=False,
            fill_kv_cache=False,
            alternate_pattern="self_cross",  # Explicitly specify the pattern
        )
        suffix_out = suffix_out[:, -self.config.n_action_steps :] # [B, 50, 1024]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out) # [B, 50, 32]

        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim) # (B, 50, 32)
            noise = self.sample_noise(actions_shape, device) # (B, 50, 32)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks) # [B, (K * 256) + 48, (K * 256) + 48]
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1 # [B, (K * 256) + 48]

        prompt_embeds = self.paligemma.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            inputs_emb=prefix_embs,
        )

        prompt_embeds = self.connector(prompt_embeds) # [B, 64, 1024]

        _, past_key_values = self.action_expert.forward(
            suffix_embeds=None,
            prompt_embeds=prompt_embeds,
            suffix_attention_mask=None,
            cross_attention_mask=None,
            suffix_position_ids=None,
            past_key_values=None,
            use_cache=True,
            fill_kv_cache=True,  # Fill the KV cache for the first step
            alternate_pattern="self_cross",  # Explicitly specify the pattern
        )
        dt = -1.0 / self.config.num_steps # -0.1
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize) # (B, )
            v_t = self.denoise_step(
                state,
                prompt_embeds,
                past_key_values,
                x_t,
                expanded_time,
            )
            # Euler step
            x_t += dt * v_t
            time += dt
        return x_t # (B, 50, 32) - actions sampled from the model

    def denoise_step(
        self,
        state,
        prompt_embeds,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_embs.shape[1] # 1 + 50
        batch_size = prompt_embeds.shape[0] # B
        prefix_len = prompt_embeds.shape[1] # 64
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks) # [B, 1 + 50, 1 + 50]
        suffix_position_ids = torch.cumsum(suffix_pad_masks, dim=1) - 1 # [B, 1 + 50]

        cross_attention_mask = torch.ones((batch_size, suffix_len, prefix_len), dtype=torch.bool, device=suffix_att_2d_masks.device)
        
        suffix_out, _ = self.action_expert.forward(
            suffix_embeds=suffix_embs,
            prompt_embeds=prompt_embeds,
            suffix_attention_mask=suffix_att_2d_masks,
            cross_attention_mask=cross_attention_mask,
            suffix_position_ids=suffix_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            fill_kv_cache=False, 
            alternate_pattern="self_cross",  # Explicitly specify the pattern
        )
        suffix_out = suffix_out[:, -self.config.n_action_steps :] # [B, 50, 1024]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out) # [B, 50, 32]
        return v_t

