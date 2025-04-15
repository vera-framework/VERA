import sys
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

import warnings
from typing import Any, List, Optional, Tuple, Union

import torch.distributed as dist
import torch.utils.checkpoint
import transformers

import sys
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging


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

# folder for saving the initial segment-level 0/1 prediction score
save_folder = "prediction_scores_InternVL2/"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Please put the guiding questions here
new_qs = "1. Are there any people in the video who are not in their typical positions or engaging in activities that are not consistent with their usual behavior?"+"\n"+ \
"2. Are there any vehicles in the video that are not in their typical positions or being used in a way that is not consistent with their usual function?"+"\n"+ \
"3. Are there any objects in the video that are not in their typical positions or being used in a way that is not consistent with their usual function?"+"\n"+ \
"4. Is there any visible damage or unusual movement in the video that indicates an anomaly?"+"\n"+ \
"5. Are there any unusual sounds or noises in the video that suggest an anomaly?"+"\n"



def build_transform(input_size, mean, std):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

def dynamic_preprocess(frames, image_size=448, grid_size=1):
    tile_size = image_size // grid_size
    num_frames = grid_size * grid_size

    assert len(frames) == num_frames, f"{num_frames} frames are required to stitch into a {grid_size}x{grid_size} grid."

    resized_frames = [frame.resize((tile_size, tile_size)) for frame in frames]
    
    stitched_image = Image.new('RGB', (image_size, image_size))

    for idx, frame in enumerate(resized_frames):
        row = idx // grid_size
        col = idx % grid_size
        stitched_image.paste(frame, (col * tile_size, row * tile_size))

    return stitched_image

class Video_Instruct_Dataset_Inference(Dataset):
    def __init__(self, vis_root, ann_root, data_type='video',sampling_rate=16, snippet_len=10):
        data_path = pathlib.Path(ann_root)
        with data_path.open(encoding='utf-8') as f:
            self.annotation = json.load(f)
        
        self.vis_root = vis_root
        self.resize_size = 448
        self.sampling_rate = sampling_rate
        self.snippet_len = snippet_len
        self.data_type = data_type
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)

        self.transform = build_transform(self.resize_size, self.IMAGENET_MEAN, self.IMAGENET_STD)
    
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        num_retries = 10  
        for _ in range(num_retries):
            try:
                sample = self.annotation[index]
                video_name = sample['video'].split('/')[-1].split('.')[0]
                video_path = os.path.join(self.vis_root, video_name)
                if 'Normal' in video_path:
                    video_label_vad = 0
                else:
                    video_label_vad = 1
                    
                vlen = sample['length']

            except:
                print(f"Failed to load examples with video: {video_path}. Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        
        return video_path, video_name, vlen

            
class VideoAnomalyDetectionModelInference(pl.LightningModule):
    def __init__(self, model, tokenizer, optimizer_instruct, model_instruct, new_qs, generation_config, epochs=5):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer_instruct = optimizer_instruct
        self.model_instruct = model_instruct
        self.new_qs = new_qs
        self.generation_config = generation_config

    def forward(self, pixel_values, num_patches_list):
        
        self.model.eval()
        batch_size = pixel_values.size(0)
        frame_size = pixel_values.size(1)
    
        pixel_values = pixel_values.to(torch.bfloat16)
 
        pixel_values = torch.reshape(pixel_values, (batch_size * frame_size, pixel_values.size(2), pixel_values.size(3), pixel_values.size(4)))

        video_prefix = ''.join([f'Frame{i + 1}: <image>\n' for i in range(len(num_patches_list))])
    
        question = self.model_instruct.replace('$Data', video_prefix)
    
        position_flag = "Based on the analysis above, please conclude your answer to 'Is there any anomaly in the video?' in 'Yes, there is an anomaly' or 'No, there is no anomaly'."
        q_start_index = question.find(position_flag)
        
        if q_start_index != -1:
            question = question[:q_start_index] + self.new_qs + question[q_start_index:]

        questions = [question] * batch_size
        num_patches_list_real = [frame_size] * batch_size

        responses = self.model.batch_chat(self.tokenizer, pixel_values, num_patches_list=num_patches_list_real, questions=questions, generation_config=generation_config)
        predict_labels = torch.tensor([1 if response[-1] == '1' else 0  for response in responses]).to(pixel_values.device)

        return predict_labels

path = 'OpenGVLab/InternVL2-8B'


model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    use_flash_attn=True,
    cache_dir='./cache'
).eval()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, cache_dir='./cache')

generation_config = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)

test_dataset = Video_Instruct_Dataset_Inference(vis_root='data/ucf/frames/', ann_root='data/UCF_Eval.json')

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)


file = open('VERA_learner_instruct.txt','r')
model_instruct =  file.read()
file.close()


lightning_model = VideoAnomalyDetectionModelInference(
    model=model, tokenizer=tokenizer, optimizer_instruct=None,
    model_instruct=model_instruct, new_qs=new_qs, generation_config=generation_config
).eval().cuda()

iter_test_loader = iter(test_loader)

fps = 30
t_window = 10
sampling_rate = 16
snippet_len = 8
image_transform = test_dataset.transform

dict_all = {}
count = -1



while True:

    dict_single ={}
    try:
        video_path, video_name, n_frms  = next(iter_test_loader)
        count+=1

        print("testing video "+video_name[0])


        if os.path.exists(save_folder+"{}.json".format(video_name[0])):
            continue
        

        start, end = 0, n_frms.item()
        vlen = n_frms.item()
        
        indices = [i for i in range(start, end, sampling_rate)]

        for ind in indices:
            snippet_start = max(ind - 1/2*fps*t_window, start)
            snippet_end = min(ind+1/2*fps*t_window, end)

            snippet_indices = np.arange(snippet_start, snippet_end, (snippet_end-snippet_start)/snippet_len).astype(int).tolist()
            frame_no_list=['{:06d}'.format(i)+'.jpg' for i in snippet_indices]
            pixel_values_list, num_patches_list = [], []
            for frame_name in frame_no_list:
    
                frame_path = video_path[0] + '/' + frame_name
                frame = Image.open(frame_path).convert('RGB')
                
                stitched_image = dynamic_preprocess([frame], grid_size=1)

                pixel_values = [image_transform(stitched_image)]
                pixel_values = torch.stack(pixel_values)

                pixel_values_list.append(pixel_values)
                num_patches_list.append(pixel_values.shape[0])

            pixel_values = torch.cat(pixel_values_list).unsqueeze(0)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()

           
                
            predict_labels = lightning_model.forward(pixel_values, num_patches_list)

            dict_single[ind]={'start':snippet_start,'end':snippet_end, 'score':predict_labels.item()}

        with open(save_folder+"{}.json".format(video_name[0]),"w") as outfile:
            
            json.dump(dict_single, outfile)

    except (StopIteration):
        break
