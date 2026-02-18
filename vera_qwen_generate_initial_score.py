import os


save_folder = "/path/to/saved/prediction/folder/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)



new_qs = "1. Are there any unusual movements or positions of the vehicles in the video frames?"+"\n"+ \
"2. Are there any inconsistencies in the actions of the vehicles in the video frames that suggest an anomaly?"+"\n"+ \
"3. Are there any unusual behaviors of the vehicles in the video frames?"+"\n"+ \
"4. Are there any inconsistencies in the weather conditions in the video frames that suggest an anomaly?"+"\n"+ \
"5. Are there any unusual objects or items in the video frames that are not in their usual positions?"+"\n"


import sys

 

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_index


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
sys.path.append('/home/muye/project/InternVL/internvl_chat')

from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging


from datasets_qwen.video_instruction_dataset_inference import dynamic_preprocess
from datasets_qwen.video_instruction_dataset_inference import Video_Instruct_Dataset_Inference
            

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

class VideoAnomalyDetectionModelInference(pl.LightningModule):
    def __init__(self, model, processor, model_instruct, new_qs, generation_config, epochs=5):
        super().__init__()
        self.model = model
        self.processor = processor
        self.model_instruct = model_instruct
        self.new_qs = new_qs
        self.generation_config = generation_config


    def forward(self, frame_path, batch_idx=0):
        
        self.model.eval()
        frame_path_list =[  frame_path[j] for j in range(len(frame_path))]
 
            
        message = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": frame_path_list,
                                "fps": 1.0,
                            },
                        ],
                    }
                ]

        
        sys_start='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
        sys_end = '<|im_end|>\n<|im_start|>assistant\n'
        video_prefix = '<|vision_start|><|video_pad|><|vision_end|>'
        question = self.model_instruct.replace('[$Data]', video_prefix)

        # Insert new questions into the prompt
        position_flag = "Based on the analysis above"
        q_start_index = question.find(position_flag)

        # Adjust question insertion logic
        if q_start_index != -1:
            question = question[:q_start_index] + self.new_qs + question[q_start_index:]
        question = sys_start+question+sys_end
        

        image_inputs, video_inputs = process_vision_info(message)
        inputs = self.processor(
            text=[question],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        responses = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        predict_labels = []
        for response in responses:
            split_response = response.split('Output')
            response = split_response[0]
            if '0' in split_response[-1]:
                predict_labels.append(0)
            else:
                predict_labels.append(1)

        return predict_labels


model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
).eval()

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")


# Create the datasets
test_dataset = Video_Instruct_Dataset_Inference(vis_root='Data/ucf/frames/', ann_root='Data/UCF_Eval.json')

# Create the data loaders
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)


file = open('VERA_learner_instruct.txt','r')
model_instruct =  file.read()
file.close()

# Initialize Lightning Model
lightning_model = VideoAnomalyDetectionModelInference(
    model=model,  processor=processor, 
    model_instruct=model_instruct, new_qs=new_qs, generation_config=None
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

        if not count in folder_test_set:
            continue

        print("testing video "+video_name[0])

        import os.path

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
            frame_path_list =[f"{video_path[0]}/{filename}" for filename in frame_no_list]
                
            predict_labels = lightning_model.forward(frame_path_list)

            dict_single[ind]={'start':snippet_start,'end':snippet_end, 'score':predict_labels.item()}

        with open(save_folder+"{}.json".format(video_name[0]),"w") as outfile:
            
            json.dump(dict_single, outfile)

    except (StopIteration):
        break
 
