import json
import pathlib
from sklearn.metrics import auc, precision_recall_curve, roc_curve
import pdb
import math
import numpy as np
from scipy.stats import iqr



ann_root = 'Data/UCF_Eval.json'
# This is the folder for loading the saved vision features
vision_folder = 'Data/vision_features/'
# This is the folder for saving the segment-level scores
score_folder = 'Data/segment_level_score/'


def gaussian_kernel_original(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) 

def gaussian_smoothing(data, sigma):
    n = len(data)
    smoothed_data = np.zeros(n)
    x = np.arange(n)

    centroid_index = int(n/2)

    kernel_values = gaussian_kernel_original(x, centroid_index, sigma)
    smoothed_data =  kernel_values * data  

    return smoothed_data


def gaussian_kernel(size, sigma):
    kernel = np.exp(-np.linspace(-size//2, size//2, size)**2 / (2*sigma**2))
    return kernel / kernel.sum() 

def gaussian_smooth_1d(data, size=5, sigma=1.0):
    kernel = gaussian_kernel(size, sigma)
    smoothed_data = np.convolve(data, kernel, mode='same')
    return smoothed_data

def softmax(row_vector):
    row_vector = row_vector
    exp_row = np.exp(row_vector - np.max(row_vector))
    
    return exp_row / np.sum(exp_row)



result = {}
import glob
import os
folder_path = './scores_77/tests/'
json_files = glob.glob(os.path.join(folder_path, '**', '*.json'), recursive=True)

for file in json_files:
    file_name = os.path.basename(file)
    with open(file,'r') as f:
        print(file_name[:-5], '=====')
        result[file_name[:-5]] = json.load(f)

auc_results = []
all_predict_score = []
all_gt = []


data_path = pathlib.Path(ann_root)
with data_path.open(encoding='utf-8') as f:
    annotation = json.load(f)

for v_i in range(len(annotation)):
    sample =  annotation[v_i]
    key = sample['video'].split('/')[-1].split('.')[0]
    print(key)

     
    score_path = score_folder +key+'.json'
    with open(score_path,'r') as f: score = json.load(f)
    result[key] = score

    
    v_array = np.load(vision_folder+key+'.npy')
    
    v_norms = np.linalg.norm(v_array, axis=1, keepdims=True)
    normalized_v = v_array / v_norms
    
    v_array = np.dot(normalized_v, normalized_v.T)

    num_segment = v_array.shape[0]
    top_n =  int(0.15*num_segment)

    v_indices = np.argsort(v_array, axis=1)[:, -top_n:]
    v_top_values = np.take_along_axis(v_array, v_indices, axis=1)

    pred_score = []
    sampling_rate = 16
    start_index = [key for key in result[key].keys()]

    all_score = []
    for i, start in enumerate(start_index):
        all_score.append(result[key][start]['score'] )

    if max(all_score) == 1.0:
        all_score = np.asarray(all_score)
        one_indices = np.where(all_score != 0)[0]
        
        all_score = all_score.tolist()


    
    score_all = []
    for i, start in enumerate(start_index):

        neighbor_score = []
        for neighbor in v_indices[i].tolist():
            neighbor_score+=[all_score[neighbor]]
        neighbor_score = np.array(neighbor_score) 
        v_softmax_values = softmax(v_top_values[i]*10) 

        v_segment_score =  np.dot(v_softmax_values, neighbor_score)
    
        score_all.append(v_segment_score)

    smoothed_data = score_all
    smoothed_data = gaussian_smooth_1d(np.array(smoothed_data), 15, 10)



    for i, start in enumerate(start_index):

        
        v_segment_score =   smoothed_data[i]

        segment_score = round(v_segment_score, 1)
        score_all.append(segment_score)
        if i != len(start_index)-1 :
            num_ele = sampling_rate
            seg_list = [segment_score]*num_ele
            pred_score.extend(seg_list)
        else:
            num_ele = int(result[key][start]['end']-int(start))
            seg_list = [segment_score]*num_ele
            pred_score.extend(seg_list)


    sigma = int(len(pred_score)/2)   
    pred_score = gaussian_smoothing(np.array(pred_score), sigma)

    gt = [0.0]*annotation[v_i]['length']
    for anno_i in range(0, len(annotation[v_i]['temporal_label']),2):
        if annotation[v_i]['temporal_label'][anno_i] != -1:
            anno_s = annotation[v_i]['temporal_label'][anno_i]
            if annotation[v_i]['temporal_label'][anno_i+1] > annotation[v_i]['length']:
                anno_e = annotation[v_i]['length']
            else:
                anno_e = annotation[v_i]['temporal_label'][anno_i+1]

            gt[anno_s:anno_e] = [1.0]*(anno_e-anno_s)


    all_predict_score.extend(pred_score)
    all_gt.extend(gt)
fpr, tpr, threshold = roc_curve(all_gt, all_predict_score)
roc_auc = auc(fpr, tpr)
   

print(roc_auc)
