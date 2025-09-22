import json
import os
from random import shuffle, Random

import numpy as np
from functools import partial

import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from torchvision.transforms import v2
from torchvision import transforms

class DataProvider(Dataset):
    # need to read whole json dataset into memory before loading to gpu
    # predict future 10 steps image
    def __init__(self, dataset_path, image_size=224, future_step=10, split="train", seed=42):
        self.dataset_path = dataset_path
        self.dinov3_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225)),
        ])
        self.image_size = image_size
        self.episodes, self.actions, self.states = load_preprocessed_data(dataset_path) 
        self.length_episodes = np.cumsum([len(i) for i in self.episodes])
        self.length_episodes = {i: self.length_episodes[i] for i in range(len(self.length_episodes))}
        self.future_step = future_step
        print("Formatting Future prediction (PRE) data")
        
        # Split dataset into train/val
        self.split = split
        self.seed = seed
        self.indices = self._generate_indices()
        
    def _generate_indices(self):
        # Generate indices for train/val split
        rng = Random(self.seed)
        all_indices = list(range(len(self.actions)))
        rng.shuffle(all_indices)
        
        # 90% train, 10% validation
        split_idx = int(0.9 * len(all_indices))
        
        if self.split == "train":
            return all_indices[:split_idx]
        elif self.split == "val":
            return all_indices[split_idx:]
        else:
            return all_indices  # Return all indices if no specific split is requested

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        # Map the index to the actual data index
        actual_index = self.indices[index]
        data_dict = self.get_raw_items(actual_index)
        future_index = self.get_future_index(actual_index, future_step=self.future_step)
        data_dict_future = self.get_raw_items(future_index)
        # assert data_dict['input_ids'] == data_dict_future['input_ids']
        data_dict['images_static_future'] = data_dict_future['images_static']
        if actual_index == future_index:
            actions = torch.tensor(self.actions[actual_index:future_index + 1])
        else:
            actions = torch.tensor(self.actions[actual_index:future_index])  # n,7
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
            image_static = self.dinov3_transform(image_static)
        else:
            raise NotImplementedError
        states = self.states[episode_idx]
        data_dict = dict(
            states=states,
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
    actions = []
    states = []
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
                states.append(frame["state"])
            else:
                raise NotImplementedError

        episodes.append(frames)
    return episodes, actions, states


def collate_fn(instances,):
    batch = {}
    batch['images_static'] = torch.stack([instance['images_static'] for instance in instances])
    batch['images_static_future'] = torch.stack([instance['images_static_future'] for instance in instances])
    batch['actions'] = torch.stack([instance['actions'] for instance in instances])
    batch['states'] = torch.stack([torch.tensor(instance['states']) for instance in instances])
    return batch


def get_future_view_prediction_w_action_data_loader(dataset_path,
                                                    batch_size,
                                                    num_workers,
                                                    world_size,
                                                    local_rank,
                                                    resolution=256,
                                                    future_step=10,
                                                    split="train",
                                                    seed=42):
    dataset = DataProvider(dataset_path, image_size=resolution, future_step=future_step, split=split, seed=seed)
    datasampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=(split=="train"))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=datasampler,
        shuffle=False,
    )
    return dataloader

def get_train_val_data_loaders(dataset_path,
                               batch_size,
                               num_workers,
                               world_size,
                               local_rank,
                               resolution=256,
                               future_step=10,
                               seed=42):
    """
    Get both training and validation data loaders with a 9:1 split.
    
    Args:
        dataset_path: Path to the dataset
        batch_size: Batch size for the data loaders
        num_workers: Number of worker processes
        world_size: Number of distributed processes
        local_rank: Rank of the current process
        resolution: Image resolution
        future_step: Number of future steps to predict
        seed: Random seed for reproducible splits
        
    Returns:
        train_loader, val_loader: Training and validation data loaders
    """
    train_loader = get_future_view_prediction_w_action_data_loader(
        dataset_path, batch_size, num_workers, world_size, local_rank,
        resolution=resolution, future_step=future_step, split="train", seed=seed
    )
    
    val_loader = get_future_view_prediction_w_action_data_loader(
        dataset_path, batch_size, num_workers, world_size, local_rank,
        resolution=resolution, future_step=future_step, split="val", seed=seed
    )
    
    return train_loader, val_loader