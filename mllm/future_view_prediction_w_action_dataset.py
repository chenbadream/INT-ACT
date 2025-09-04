import json
import os
from random import shuffle

import numpy as np
from functools import partial

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from torchvision import transforms

def image_transform(image, resolution=256, normalize=True):
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    return image

class DataProvider(Dataset):
    # need to read whole json dataset into memory before loading to gpu
    # predict future 10 steps image
    def __init__(self, dataset_path, image_size=512, future_step=10):
        self.dataset_path = dataset_path
        self.transform = image_transform
        self.image_size = image_size
        self.episodes, self.instructions, self.actions = load_preprocessed_data(dataset_path)  # actions are not used
        self.length_episodes = np.cumsum([len(i) for i in self.episodes])
        self.length_episodes = {i: self.length_episodes[i] for i in range(len(self.length_episodes))}
        self.future_step = future_step
        print("Formatting Future prediction (PRE) data")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        data_dict = self.get_raw_items(index)
        future_index = self.get_future_index(index, future_step=self.future_step)
        data_dict_future = self.get_raw_items(future_index)
        # assert data_dict['input_ids'] == data_dict_future['input_ids']
        data_dict['images_static_future'] = data_dict_future['images_static']
        if index == future_index:
            actions = torch.tensor(self.actions[index:future_index + 1])
        else:
            actions = torch.tensor(self.actions[index:future_index])  # n,7
        if actions.shape[0] < self.future_step:
            offset = self.future_step - actions.shape[0]
            pad_tube = torch.zeros(size=(offset, actions.shape[-1]), dtype=actions.dtype)
            pad_tube[:, -1] = actions[-1, -1]  # gripper state of last action is repeated
            actions = torch.cat([actions, pad_tube], dim=0)
        data_dict['actions'] = actions  # (self.future_step, 7) (10,7)
        return data_dict

    def get_raw_items(self, index):
        episode_idx, idx = self.get_episode_idx(index)
        episode = self.episodes[episode_idx]
        # sequence_length * epi[0],epi[1],...
        if 'bridge' in self.dataset_path:
            image_path = episode[idx]
            image_static = Image.open(image_path).convert('RGB')
            image_static = self.transform(image_static, resolution=self.image_size)
        else:
            raise NotImplementedError
        instruction = self.instructions[episode_idx]
        data_dict = dict(
            input_ids=instruction,
            images_static=image_static,
        )
        return data_dict

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


def load_preprocessed_data(dataset_path):
    episodes = []
    instructions = []
    actions = []
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
                actions.append(frame["action"])
            else:
                raise NotImplementedError

        instructions.append(epi["instruction"])
        episodes.append(frames)
    return episodes, instructions, actions


def collate_fn(instances,):
    input_ids = [instance["input_ids"] for instance in instances]
    batch = dict(input_ids=input_ids,)
    batch['images_static'] = torch.stack([instance['images_static'] for instance in instances])
    batch['images_static_future'] = torch.stack([instance['images_static_future'] for instance in instances])
    batch['actions'] = torch.stack([instance['actions'] for instance in instances])
    return batch


def get_future_view_prediction_w_action_data_loader(dataset_path,
                                                    batch_size,
                                                    num_workers,
                                                    world_size,
                                                    local_rank,
                                                    resolution=512,
                                                    future_step=10):
    train_dataset = DataProvider(dataset_path, image_size=resolution, future_step=future_step)
    datasampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=datasampler,
        shuffle=False,
    )
    return dataloader
