import os
import torch.multiprocessing
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tqdm
import json
import cv2

def preprocess_dataset(origin_dataset_path, episode_num=10):
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    
    processed_root = "/vast/bc4227/datasets/bridge_processed"
    os.makedirs(processed_root, exist_ok=True)
    
    builder = tfds.builder_from_directory(origin_dataset_path)
    episode_ds = builder.as_dataset(split='train')
    
    dataset_info = []
    print('------processing------')
    
    processed_episode_count = 0

    for i, episode in enumerate(tqdm.tqdm(iter(episode_ds))):
        language_instruction = ""
        for step in episode['steps'].as_numpy_iterator():
            if 'language_instruction' in step and step['language_instruction']:
                language_instruction = step['language_instruction'].decode("utf-8")
                break 
        
        if not language_instruction:
            print(f"Skipping episode {i} due to missing language instruction.")
            continue
        
        episode_path = os.path.join(processed_root, f"episode{processed_episode_count:07}")
        os.makedirs(episode_path, exist_ok=True)
        
        episode_info = {
            'instruction': language_instruction,
            'frames': []
        }

        for j, step in enumerate(episode['steps'].as_numpy_iterator()):
            frame = step['observation']['image_0']
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_path = os.path.join(episode_path, f"frame{j:03}.jpg")
            cv2.imwrite(frame_path, frame)
            
            frame_info = {
                'dir': f"episode{processed_episode_count:07}/frame{j:03}.jpg",
                'action': step['action'].tolist()
            }
            episode_info['frames'].append(frame_info)
        
        dataset_info.append(episode_info)
        processed_episode_count += 1
    
    with open(os.path.join(processed_root, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)


preprocess_dataset("/vast/bc4227/datasets/bridge_dataset/1.0.0")  # change the path to your own raw bridge dataset

# python process_bridge.py
