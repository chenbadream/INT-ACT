import copy
import glob
import io
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import transformers
import yaml
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import transforms
from datasets import load_dataset, concatenate_datasets
from blip3o.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from blip3o.utils import rank0_print


ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_preprocessed_data(dataset_path):
    """Load preprocessed data from the dataset path"""
    episodes = []
    instructions = []
    assert "processed" in dataset_path
    with open(os.path.join(dataset_path, 'dataset_info.json'), 'r') as f:
        dataset = json.load(f)
    for epi in dataset:
        frames = []
        for frame in epi["frames"]:
            if 'bridge' in dataset_path:
                image_path = frame["dir"]
                full_image_path = os.path.join(dataset_path, image_path)
                frames.append(full_image_path)

        instructions.append(epi["instruction"])
        episodes.append(frames)
    return episodes, instructions

def preprocess_multimodal(sources: Sequence[str], data_args) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            replace_token = DEFAULT_IMAGE_TOKEN
            # NOTE: only add im_start_end when image generation
            if data_args.mm_use_im_start_end and sentence['from'] == 'gpt':
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    if 'image_token_index' not in globals():
        tokenizer.add_tokens(["<image>"], special_tokens=True)
        global image_token_index
        image_token_index = tokenizer.convert_tokens_to_ids("<image>")

    im_start, im_end = tokenizer.additional_special_tokens_ids[:2]
    unmask_tokens_idx =  [198, im_start, im_end]

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += input_id

        for conv in source:
            # Make sure blip3o data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += encode_id
            else:
                target += encode_id
        
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  
        labels=targets,  
    )

class FuturePredictionDataset(Dataset):
    """Dataset for future view prediction with supervised fine-tuning structure."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_path: str,
        data_args,
        future_step=10
    ):
        super(FuturePredictionDataset, self).__init__()
        
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.future_step = future_step
        self.modality = torch.tensor(1)
        
        # Load preprocessed data
        self.episodes, self.instructions = load_preprocessed_data(data_path)
        self.length_episodes = np.cumsum([len(i) for i in self.episodes])
        self.length_episodes = {i: self.length_episodes[i] for i in range(len(self.length_episodes))}
        
        # Calculate total number of frames
        total_frames = sum(len(episode) for episode in self.episodes)
        
        rank0_print(f"Loaded future prediction dataset with {total_frames} samples")

    def __len__(self):
        return sum(len(episode) for episode in self.episodes)

    def get_episode_idx(self, index):
        for i, x in self.length_episodes.items():
            if index < x:
                episode_idx = i
                idx = index - self.length_episodes[episode_idx - 1] if i != 0 else index
                return episode_idx, idx
        raise ValueError(f"Index {index} out of range")

    def get_future_index(self, index, future_step=10):
        for i, x in self.length_episodes.items():
            if index < x:
                if index + future_step < x:
                    return index + future_step  # future index is in the same episode
                else:
                    return self.length_episodes[i] - 1  # future index is in the next episode, use the last frame
        raise ValueError(f"Index {index} out of range")

    def get_raw_items(self, index):
        """Get raw image and instruction for a given index"""
        episode_idx, idx = self.get_episode_idx(index)
        episode = self.episodes[episode_idx]
        
        if 'bridge' in self.data_args.data_path:
            image_path = episode[idx]
            image_static = Image.open(image_path).convert('RGB')
        else:
            raise NotImplementedError
            
        instruction = self.instructions[episode_idx]
        
        if 'bridge' in self.data_args.data_path:
            return {
                'instruction': instruction,
                'images_static': image_static,
            }

    def process_image(self, image):
        processor = self.data_args.image_processor
        image_size = image.size
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, self.modality

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        i = 0
        # Get current timestep data
        current_data = self.get_raw_items(i)
        
        # Get future timestep data
        future_index = self.get_future_index(i, future_step=self.future_step)
        future_data = self.get_raw_items(future_index)

        # Create conversation for the model
        instruction = current_data['instruction']
        conversations = [
            {
                "from": "human", 
                "value": f"<image>\nGiven this current view and the instruction '{instruction}', predict what the scene will look like in {self.future_step} steps."
            },
            {
                "from": "gpt", 
                "value": "<image>"
            }
        ]
        
        # Process with Qwen tokenizer
        sources = preprocess_multimodal(copy.deepcopy([conversations]), self.data_args)
        data_dict = preprocess_qwen(sources, self.tokenizer, has_image=True)
        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        
        # Process current image
        current_image_processed = self.process_image(current_data['images_static'])
        
        # Process future image (same processing as current image for tokenization)
        future_image_processed = self.process_image(future_data['images_static'])
        
        # Add images to data dict - both images will be tokenized as part of the sequence
        data_dict["image"] = [current_image_processed, future_image_processed]
        data_dict["ids"] = f"future_pred_{i}"
        
        return data_dict


@dataclass
class FuturePredictionDataCollator(object):
    """Collate examples for future prediction supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
            
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        batch = dict(
            input_ids=input_ids, 
            labels=labels.long() if labels.dtype == torch.int32 else labels, 
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        )
        
        # Handle images (both current and future images are tokenized)
        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]
            batch["images"] = images

        return batch


def get_dataset_cls(name):
    if name == 'future_prediction':
        dataset_cls = FuturePredictionDataset
    else:
        raise ValueError(f'Unknown dataset class {name}')
    return dataset_cls


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for future prediction supervised fine-tuning."""
    dataset_cls = get_dataset_cls(data_args.dataset_cls)
    train_dataset = dataset_cls(
        tokenizer=tokenizer, 
        data_path=data_args.data_path, 
        data_args=data_args,
        future_step=getattr(data_args, 'future_step', 10)
    )
    data_collator = FuturePredictionDataCollator(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
