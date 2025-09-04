import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoTokenizer

from configuration_mllm import MLLMConfig
from paligemma import (
    PaliGemmaConfig,
    PaliGemmaModel,
)
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.utils import get_safe_dtype

from modeling_magvitv2 import MAGVITv2

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

    # Ensure both masks are on the same device
    device = pad_masks.device
    att_masks = att_masks.to(device)
    
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks

class MLLMPolicy(PreTrainedPolicy):
    """Wrapper class around PI0FlowMatching model to train and run inference within LeRobot."""

    config_class = MLLMConfig
    name = "mllm"

    def __init__(
        self,
        config: MLLMConfig,
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

        self.language_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

        self.vq_model = MAGVITv2.from_pretrained("showlab/magvitv2")
        self.vq_model.eval()
        self.vq_model.requires_grad_(False)

        self.model = PI0FlowMatching(config)

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()
        
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """简单实现select_action方法以满足抽象类的要求"""
        # 返回一个空张量作为占位符
        return torch.zeros(1)

    @torch.no_grad
    def sample_image(self, current_image, language_instruction) -> Tensor:
        """
        Generate future image prediction based on current image and language instruction.
        
        Args:
            current_image: Tensor of shape [C, H, W] or [1, C, H, W] representing the current image
            language_instruction: String representing the language instruction
        Returns:
            Tensor of generated future image
        """
        # Ensure the image has batch dimension
        if current_image.dim() == 3:
            current_image = current_image.unsqueeze(0)
            
        # Get image tokens
        image_static_tokens = self.vq_model.get_code(current_image) + 257216
        img_masks = torch.ones(image_static_tokens.shape[0], image_static_tokens.shape[1], 
                              dtype=torch.bool, device=image_static_tokens.device)
        
        # Process language instruction
        language_instruction = language_instruction if language_instruction.endswith("\n") else f"{language_instruction}\n"
        tokenized_prompt = self.language_tokenizer.__call__(
            [language_instruction],  # Wrap in list for batch processing
            padding="max_length",
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
            truncation=True,
        )
        lang_tokens = tokenized_prompt["input_ids"].to(image_static_tokens.device)
        lang_masks = tokenized_prompt["attention_mask"].to(dtype=torch.bool, device=image_static_tokens.device)

        predicted_tokens = self.model.generate_image(image_static_tokens, lang_tokens, img_masks, lang_masks)
        predicted_image -= 257216

        predicted_image = self.vq_model.decode_code(predicted_tokens)
        
        # Move tensors to CPU and convert to numpy
        current_image = current_image.cpu().squeeze(0)
        predicted_image = predicted_image.cpu().squeeze(0)

        predicted_image = (predicted_image + 1.0) / 2.0
        predicted_image = predicted_image.permute(1, 2, 0).numpy()

        return predicted_image

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """Do a full training forward pass to compute the loss"""

        image_stastic_tokens, image_stastic_future_tokens, img_masks = self.prepare_images(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)

        loss_dict = {}
        loss = self.model.forward(image_stastic_tokens, image_stastic_future_tokens, img_masks, lang_tokens, lang_masks)

        loss_dict["l2_loss"] = loss
        return loss, loss_dict

    def prepare_images(self, batch):
        images_stastic = batch['images_static']
        # Get the consistent device to use
        device = images_stastic.device
        
        # Get the original tokens before adding offset
        raw_image_stastic_tokens = self.vq_model.get_code(images_stastic)
        
        image_stastic_tokens = raw_image_stastic_tokens + 257216
        img_masks = torch.ones(image_stastic_tokens.shape[0], image_stastic_tokens.shape[1], dtype=torch.bool, device=device)

        images_stastic_future = batch['images_static_future'].to(device)
        
        # Check future image tokens as well
        raw_image_stastic_future_tokens = self.vq_model.get_code(images_stastic_future)
        
        image_stastic_future_tokens = raw_image_stastic_future_tokens + 257216
        
        # Ensure consistent device for all outputs
        image_stastic_tokens = image_stastic_tokens.to(device)
        image_stastic_future_tokens = image_stastic_future_tokens.to(device)
        
        return image_stastic_tokens, image_stastic_future_tokens, img_masks

    @torch.compiler.disable(recursive=False)
    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Tokenize the text input"""
        device = batch['images_static'].device
        tasks = batch["input_ids"]
        # print("tasks:", tasks)
        # PaliGemma prompt has to end with a new line
        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding="max_length",
            padding_side="left",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
            truncation=True, # Irving: add truncation to follow Allen
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)


        return lang_tokens, lang_masks

class PI0FlowMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        paligemma_config = PaliGemmaConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            paligemma_pretrained_path=self.config.paligemma_pretrained_path,
            attention_implementation=self.config.attention_implementation,
        )
        self.paligemma = PaliGemmaModel(paligemma_config)

        current_vocab_size = self.paligemma.paligemma.config._vocab_size
        new_total_vocab_size = current_vocab_size + config.codebook_size
        self.paligemma.paligemma.resize_token_embeddings(new_total_vocab_size)
        self.new_total_vocab_size = new_total_vocab_size

        self.vq_head = nn.Linear(2048, new_total_vocab_size)

    def embed_prefix(
        self, image_stastic_tokens, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        lang_emb = self.paligemma.embed_language_tokens(lang_tokens)

        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        img_emb = self.paligemma.embed_language_tokens(image_stastic_tokens)

        # Normalize image embeddings
        img_emb_dim = img_emb.shape[-1]
        img_emb = img_emb * math.sqrt(img_emb_dim)

        embs.append(img_emb)
        pad_masks.append(img_masks)

        # Create attention masks so that image tokens attend to each other
        num_img_embs = img_emb.shape[1]
        att_masks += [1] + [0] * (num_img_embs - 1)
        bsize = lang_emb.shape[0]

        # Ensure all tensors are on the same device before concatenation
        device = embs[0].device
        embs = [e.to(device) for e in embs]
        pad_masks = [m.to(device) for m in pad_masks]
        
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def forward(
        self, image_stastic_tokens, image_stastic_future_tokens, img_masks, lang_tokens, lang_masks
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        # Ensure all inputs are on the same device
        device = image_stastic_tokens.device
        image_stastic_tokens = image_stastic_tokens.to(device)
        image_stastic_future_tokens = image_stastic_future_tokens.to(device)
        img_masks = img_masks.to(device)
        lang_tokens = lang_tokens.to(device)
        lang_masks = lang_masks.to(device)
        
        labels = image_stastic_future_tokens #(b, num_vq_tokens)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            image_stastic_tokens, img_masks, lang_tokens, lang_masks
        )

        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        out_emb = self.paligemma.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            inputs_emb=prefix_embs,
        )
        future_prediction = out_emb[:, -self.config.num_vq_tokens:]
        # Original openpi code, upcast attention output
        future_prediction = future_prediction.to(dtype=torch.float32)
        logits = self.vq_head(future_prediction) #(b, num_vq_tokens, codebook_size)

        loss = F.cross_entropy(logits.view(-1, self.new_total_vocab_size), labels.view(-1))
        return loss
    
    @torch.no_grad
    def generate_image(self, image_static_tokens, lang_tokens, img_masks, lang_masks) -> Tensor:     
        # Embed prefix
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            image_static_tokens, img_masks, lang_tokens, lang_masks
        )
        
        # Create attention masks
        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        
        # Forward pass through the model
        out_emb = self.paligemma.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            inputs_embeds=prefix_embs,
        )
        
        # Extract future prediction
        future_prediction = out_emb[:, -self.config.num_vq_tokens:]
        future_prediction = future_prediction.to(dtype=torch.float32)
        logits = self.vq_head(future_prediction)
        
        # Get the most likely tokens
        predicted_tokens = torch.argmax(logits, dim=-1)
        
        return predicted_tokens