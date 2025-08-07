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

from datasets.video_instruction_dataset_VERA import Video_Instruct_Dataset
import math
import torch
from transformers import AutoTokenizer, AutoModel

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    print(world_size, 'xxx')
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

class VideoAnomalyDetectionModel(pl.LightningModule):
    def __init__(self, model, tokenizer, optimizer_instruct, model_instruct, new_qs, generation_config, epochs=5):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer_instruct = optimizer_instruct
        self.model_instruct = model_instruct
        self.new_qs = new_qs
        self.generation_config = generation_config
        self.epochs = epochs
        self.validation_step_outputs = []
        self.validation_step_count = []
        self.automatic_optimization = False

    def forward(self, pixel_values, num_patches_list, batch_idx=0):
        
        self.model.eval()
        # Get batch and frame size
        batch_size = pixel_values.size(0)
        frame_size = pixel_values.size(1)
    
        # Reshape pixel_values for processing
        pixel_values = pixel_values.to(torch.bfloat16)
 
        pixel_values = torch.reshape(pixel_values, (batch_size * frame_size, pixel_values.size(2), pixel_values.size(3), pixel_values.size(4)))
    
        # Create video frame prefix for each frame
        video_prefix = ''.join([f'Frame{i + 1}: <image>\n' for i in range(len(num_patches_list))])
    
        # Insert the reshaped prefix into the question
        question = self.model_instruct.replace('$Data', video_prefix)
    
        # Insert new questions into the prompt
        position_flag = "Based on the analysis above"
        q_start_index = question.find(position_flag)
        
        # Adjust question insertion logic
        if q_start_index != -1:
            question = question[:q_start_index] + self.new_qs + question[q_start_index:]
    
        # Prepare the list of questions, one for each batch
        questions = [question] * batch_size
        num_patches_list_real = [frame_size] * batch_size

        # Generate responses from the model
        responses = self.model.batch_chat(self.tokenizer, pixel_values, num_patches_list=num_patches_list_real, questions=questions, generation_config=generation_config)
        
        if batch_idx % 10 == 0:
            f=open('./log_generating_5q.txt','a')
            f.write('1--------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
            f.write(question+'\n')
            f.write('2--------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')

            f.write('\n')
            f.close()
        return responses, pixel_values, num_patches_list_real

    def training_step(self, batch, batch_idx):

        self.model.eval()
        
        pixel_values, num_patches_list, labels, video_name = batch

        # Get batch and frame size
        batch_size = pixel_values.size(0)
        frame_size = pixel_values.size(1)

        responses, pixel_values, num_patches_list_real = self.forward(pixel_values, num_patches_list, batch_idx)

        # Convert the last token in response to a label (0 or 1)
        predict_labels = []
        for response in responses:
            split_response = response.split('Output')
            response = split_response[0]
            if '0' in split_response[-1]:
                predict_labels.append(0)
            else:
                predict_labels.append(1)
        
        
        # Modify the optimizer instructions by replacing placeholders with predictions and ground truth
        optimizer_instruct_batch = self.optimizer_instruct.replace(
            '[[$Prediction] [$Prediction] [$Prediction] [$Prediction] [$Prediction] [$Prediction] [$Prediction] [$Prediction] [$Prediction] [$Prediction]]',
            str(predict_labels)
        )
    
        optimizer_instruct_batch = optimizer_instruct_batch.replace(
            '[[$GroundTruth] [$GroundTruth] [$GroundTruth] [$GroundTruth] [$GroundTruth] [$GroundTruth] [$GroundTruth] [$GroundTruth] [$GroundTruth] [$GroundTruth]]',
            str(labels.tolist())
        )
    
        # Insert the input video prefix into the optimizer instructions
        position_flag = "** Model Descriptions: **"
        input_start_index = optimizer_instruct_batch.find(position_flag)

        # Create video frame prefix for each frame
        video_prefix = ''.join([f'Frame{i + 1}: <image>\n' for i in range(frame_size)])
        
        input_string = '[' + ('[' + video_prefix + '] ') * batch_size + ']\n'
        optimizer_instruct_batch = optimizer_instruct_batch[:input_start_index] + input_string + optimizer_instruct_batch[input_start_index-1:]
    
        # Insert the current prompt question into the optimizer instructions
        prompt_question = self.new_qs
        position_flag = "Based on the analysis above"
        q_start_index = optimizer_instruct_batch.find(position_flag)
        complete_optimizer_instruct = optimizer_instruct_batch[:q_start_index] + self.new_qs + optimizer_instruct_batch[q_start_index:]

        
        # Generate model response for training
        response = self.model.chat(tokenizer, pixel_values, complete_optimizer_instruct, generation_config)

 
        # get new questions
        if response.find('New Prompt Questions') == -1:
            pass
        else:
            self.new_qs = response[response.find('New Prompt Questions')+len('New Prompt Questions')+2:]
            if len(self.new_qs.split('\n'))>5:
                candidate=self.new_qs.split('\n')[:5]
                self.new_qs = candidate[0]+'\n'+candidate[1]+'\n'+candidate[2]+'\n'+candidate[3]+'\n'+candidate[4]+'\n'

        # Convert the last token in response to a label (0 or 1)
        predict_labels = []
        for response in responses:
            split_response = response.split('Output')
            response = split_response[0]
            if '0' in split_response[-1]:
                predict_labels.append(0)
            else:
                predict_labels.append(1)
        
        predict_labels = torch.tensor(predict_labels).to(pixel_values.device)
        
        # Calculate accuracy by comparing predicted labels to actual labels
        correct_predictions = (predict_labels == labels).sum()
        accuracy = correct_predictions / len(labels)

        # Log validation accuracy
        self.log('train_acc', accuracy.item(), prog_bar=True, on_step=True, sync_dist=True, batch_size=len(labels))

        if batch_idx % 10 == 0:
            f=open('./log_generating_5q.txt','a')
            f.write('1--------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
            f.write(complete_optimizer_instruct+'\n')
            f.write('2--------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')

            f.write('\n')
            f.close()

            f=open('./log_generating_5q.txt','a')
            f.write('1========================================================================================================================================================================\n')
            f.write(response+'\n')
            f.write('2========================================================================================================================================================================\n')
            f.write('\n')
            f.close()
            
        
        return torch.tensor(1.0, requires_grad=True).to(pixel_values.device) 

     
    def validation_step(self, batch, batch_idx):
        pixel_values, num_patches_list, labels, video_name = batch
        responses, pixel_values, num_patches_list_real = self.forward(pixel_values, num_patches_list)

        # Convert the last token in response to a label (0 or 1)
        predict_labels = []
        for response in responses:
            split_response = response.split('Output')
            response = split_response[0]
            if '0' in split_response[-1]:
                predict_labels.append(0)
            else:
                predict_labels.append(1)
        
        predict_labels = torch.tensor(predict_labels).to(pixel_values.device)

        # Calculate accuracy by comparing predicted labels to actual labels
        correct_predictions = (predict_labels == labels).sum()
        accuracy = correct_predictions / len(labels)
        
        # Log validation accuracy
        self.log('val_acc', accuracy.item(), prog_bar=True, on_step=True, sync_dist=True, batch_size=len(labels))

        self.validation_step_outputs.append(correct_predictions)
        self.validation_step_count.append(len(labels))

        epoch_average = torch.stack(self.validation_step_outputs).sum()/sum(self.validation_step_count)
        print(epoch_average.item())

        return {'val_acc': accuracy}
    

    def on_validation_epoch_end(self):
        # Calculate average accuracy across all batches in the validation set
        epoch_average = torch.stack(self.validation_step_outputs).sum()/sum(self.validation_step_count)
        self.log("val_avg_acc", epoch_average.item(), sync_dist=True)
        self.validation_step_outputs.clear()  # free memory
        self.validation_step_count.clear()  # free memory

        f=open(text_file, 'a')
        f.write(f'accuracy: {epoch_average.item()}'+'\n')
        f.write('New Prompt Questions:'+'\n')
        f.write(self.new_qs+'\n')
        f.write('\n')
        f.close()
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

text_file = './VERA_generated_question.txt'

f = open(text_file, 'w')
f.writelines('Generative Questions\n')
f.close()

path = 'OpenGVLab/InternVL2-8B'
 
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    # load_in_8bit=True,
    cache_dir='./cache',
    # device_map=device_map
    ).eval()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False, cache_dir='./cache')

generation_config = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)

# Create the datasets; please put the frames of XD Violence in the folder Data/ucf/
train_dataset = Video_Instruct_Dataset(vis_root='Data/ucf/', ann_root='Data/UCF_Instruct_train.json', num_sampled_frame=8)
val_dataset = Video_Instruct_Dataset(vis_root='Data/ucf/', ann_root='Data/UCF_Instruct_val.json', TEST_FLAG=True, num_sampled_frame=8)

# Create the datasets for XD-Violence; please put the frames of XD Violence in the folder Data/xd_violence/
# train_dataset = Video_Instruct_Dataset(vis_root='Data/xd_violence/', ann_root='Data/xd_train.json', num_sampled_frame=8)
# val_dataset = Video_Instruct_Dataset(vis_root='Data/xd_violence/', ann_root='Data/xd_val.json', TEST_FLAG=True, num_sampled_frame=8)


# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=16, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=1, drop_last=False)

new_qs = '1. Is there any suspicious person or object that looks unusual in this scene?'+'\n' + '2. Is there any behavior that looks unusual in this scene?'+'\n'

file = open('VERA_learner_instruct.txt','r')
model_instruct =  file.read()
file.close()

file = open('VERA_optimizer_instruct.txt','r')
optimizer_instruct =  file.read()
file.close()


# Initialize Lightning Model
lightning_model = VideoAnomalyDetectionModel(
    model=model, tokenizer=tokenizer, optimizer_instruct=optimizer_instruct,
    model_instruct=model_instruct, new_qs=new_qs, generation_config=generation_config
)

from pytorch_lightning.loggers import TensorBoardLogger
tb_logger = TensorBoardLogger(save_dir="logs_VERA/")

# Train the model using the Lightning Trainer
trainer = pl.Trainer(
    logger=tb_logger,
    val_check_interval=100,
    log_every_n_steps=5,
    max_epochs=10, 
    enable_checkpointing=False,
    devices=1,  # Use all available GPUs
    accelerator="gpu",  # GPU training
    strategy="ddp"  # Distributed Data Parallel training
)


TRAIN_FLAG=True

if TRAIN_FLAG:
    trainer.fit(lightning_model, train_loader, val_loader)
else:
    trainer.validate(lightning_model, val_loader)
