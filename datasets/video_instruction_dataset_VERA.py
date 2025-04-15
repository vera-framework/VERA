import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import decord
from decord import VideoReader, cpu
import random
import torch
from torch.utils.data.dataloader import default_collate
from PIL import Image
from typing import Dict, Optional, Sequence
import transformers
import pathlib
import json
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import copy
import math
from torchvision import transforms
import pdb
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel
import pytorch_lightning as pl
import itertools


def build_transform(input_size, mean, std):

    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

def dynamic_preprocess(frames, image_size=448, grid_size=1):
    """
    Takes grid_size*grid_size frames and stitches them into a single image of size `image_size`.
    
    Args:
    - frames: List of frames to stitch into a grid.
    - image_size: Final size of the stitched image (default 448x448).
    - grid_size: Number of frames per row and column (default 2x2).

    Returns:
    - stitched_image: A stitched image of size `image_size` with the frames arranged in a grid.
    """
    tile_size = image_size // grid_size
    num_frames = grid_size * grid_size

    assert len(frames) == num_frames, f"{num_frames} frames are required to stitch into a {grid_size}x{grid_size} grid."

    # Resize each frame to fit within the grid
    resized_frames = [frame.resize((tile_size, tile_size)) for frame in frames]
    
    # Create a blank canvas for the final stitched image
    stitched_image = Image.new('RGB', (image_size, image_size))

    # Paste each frame into the grid
    for idx, frame in enumerate(resized_frames):
        # Calculate row and column for each frame based on grid_size
        row = idx // grid_size
        col = idx % grid_size
        stitched_image.paste(frame, (col * tile_size, row * tile_size))

    return stitched_image

class Video_Instruct_Dataset(Dataset):
    def __init__(self, vis_root, ann_root, num_sampled_frame, num_video_query_token=32, data_type='video', TEST_FLAG=False):
        data_path = pathlib.Path(ann_root)
        with data_path.open(encoding='utf-8') as f:
            self.annotation = json.load(f)
        
        self.num_video_query_token = num_video_query_token
        self.vis_root = vis_root
        self.resize_size = 448
        self.num_frm = 16
        self.num_seg = 4
        self.grid_size = 1
        self.num_frm_seg = 2*self.grid_size*self.grid_size
        self.max_num = 1
        self.data_type = data_type
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.transform = build_transform(self.resize_size, self.IMAGENET_MEAN, self.IMAGENET_STD)
        self.test_flag = TEST_FLAG

        self.num_sampled_frame = num_sampled_frame
    
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            try:
                sample = self.annotation[index]
                video_name = sample['video'].split('/')[-1].split('.')[0]
                video_path = os.path.join(self.vis_root, 'frames', video_name)
                if 'Normal' in video_path:
                    video_label_vad = 0
                else:
                    video_label_vad = 1
                conversation_list = sample['QA']

                vlen = sample['length']
                start, end = 0, vlen
 
                indices = []
                
                random_integers = np.arange(start, end, (end - start) / self.num_sampled_frame).astype(int).tolist() 
                sorted_integers = np.sort(random_integers)
                indices += sorted_integers.tolist()

                # Pick frame indices and create a list of 4-frame groups
                frame_no_list = ['{:06d}'.format(i) + '.jpg' for i in indices]
                frame_groups = [frame_no_list[i:i + self.grid_size*self.grid_size] for i in range(0, len(frame_no_list), self.grid_size*self.grid_size)]

                pixel_values_list = []  # List to store pixel values for all stitched images
                num_patches_list = []   # Track the number of patches per stitched image

                for frame_group in frame_groups:
                    if len(frame_group) < self.grid_size*self.grid_size:  # Ensure we have exactly grid_size*grid_size frames in each group
                        continue

                    # Load and preprocess the 4 frames
                    frames = []
                    for frame_idx in frame_group:
                        frame_path = video_path + '/' + frame_idx
                        frame = Image.open(frame_path).convert('RGB')
                        frames.append(frame)

                    # Stitch the grid_size*grid_size frames into a single 448x448 image
                    stitched_image = dynamic_preprocess(frames, grid_size=self.grid_size)

                    # Apply transform to the stitched image
                    pixel_values = [self.transform(stitched_image)]
                    pixel_values = torch.stack(pixel_values)

                    # Store the pixel values and number of patches
                    pixel_values_list.append(pixel_values)
                    num_patches_list.append(pixel_values.shape[0])

            except:
                print(f"Failed to load examples with video: {video_path}. Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

        # Concatenate pixel values for all stitched images
        pixel_values = torch.cat(pixel_values_list)
        
        return pixel_values, num_patches_list, video_label_vad, video_name
