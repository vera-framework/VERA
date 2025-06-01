import json
import pathlib
from sklearn.metrics import auc, precision_recall_curve, roc_curve
import pdb
import math
import numpy as np
from scipy.stats import iqr
import os

def gaussian_kernel_original(x, mu, sigma):
    """Compute the Gaussian kernel."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) 

def gaussian_smoothing(data, sigma):
    """Apply Gaussian smoothing to a 1-D array with centroid at index 25."""
    n = len(data)
    smoothed_data = np.zeros(n)
    x = np.arange(n)
  
    centroid_index = int(n/2)

    kernel_values = gaussian_kernel_original(x, centroid_index, sigma)
    smoothed_data =  kernel_values * data  

    return smoothed_data


def gaussian_kernel(size, sigma):
    """Generate a Gaussian kernel."""
    kernel = np.exp(-np.linspace(-size//2, size//2, size)**2 / (2*sigma**2))
    return kernel / kernel.sum()  # Normalize the kernel

def gaussian_smooth_1d(data, size=5, sigma=1.0):
    """Apply Gaussian smoothing to 1D data."""
    kernel = gaussian_kernel(size, sigma)
    smoothed_data = np.convolve(data, kernel, mode='same')
    return smoothed_data

def softmax(row_vector):
    # Subtract the max for numerical stability
    row_vector = row_vector
    exp_row = np.exp(row_vector - np.max(row_vector))
    
    return exp_row / np.sum(exp_row)

ann_root = 'path/to/Data/xd_annotations.json'
score_folder = 'path/to/XD-Violence/scores'
vision_folder = 'path/to/XD-Violence/vision_features/'
 
result = {}

auc_results = []
all_predict_score = []
all_gt = []
 

data_path = pathlib.Path(ann_root)
with data_path.open(encoding='utf-8') as f:
    annotation = json.load(f)

with open('path/to/Data/name_gt_pair.json') as f:
    gt_order=json.load(f)

for v_i in range(len(annotation)):
    sample =  annotation[v_i]
    key = sample['video']
    # this line is to ensure we just test on test videos in XD-Violence
    video_path = 'path/to/XD-Violence/videos/'+key+'.mp4'
    if not os.path.exists(video_path):
        continue
    print(key)
 

    gt = gt_order[key]

 
    score_path = score_folder +key+'.json'
    if not os.path.exists(score_path):
        continue

    with open(score_path,'r') as f: score = json.load(f)
    result[key] = score
 
    v_array = np.load(vision_folder+key+'.npy')
   
    v_norms = np.linalg.norm(v_array, axis=1, keepdims=True)
    normalized_v = v_array / v_norms
    
    v_array = np.dot(normalized_v, normalized_v.T)

    num_segment = len(result[key].keys())

    if num_segment > 4:
        top_n =  int(0.15*num_segment)
    else:
        top_n = num_segment
 
    v_array = 1*v_array
    v_indices = np.argsort(v_array, axis=1)[:, -top_n:]
    v_top_values = np.take_along_axis(v_array, v_indices, axis=1)

    pred_score = []
    sampling_rate = 16
    start_index = [key for key in result[key].keys()]

    all_score = []
    for i, start in enumerate(start_index):
        all_score.append(result[key][start]['score'] )
    
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

        
        segment_score = v_segment_score
        score_all.append(segment_score)
        if i != len(start_index)-1 :
            num_ele = sampling_rate
            seg_list = [segment_score]*num_ele
            pred_score.extend(seg_list)
        else:
            num_ele = int(result[key][start]['end']-int(start))
            seg_list = [segment_score]*num_ele
            pred_score.extend(seg_list)

    pred_score=1/(1+np.exp(-np.array(pred_score)) )
    pred_score =  [ round(num,1)  for num in pred_score]

    lens = len(pred_score)
    mod = (lens-1) % sampling_rate # minusing 1 is to align flow  rgb: minusing 1 when extracting features
    pred_score = pred_score[:-1]
    if mod:
        pred_score = pred_score[:-mod]

    all_predict_score.extend(pred_score)
    all_gt.extend(gt)
 

precision, recall, th = precision_recall_curve(all_gt, all_predict_score)
pr_auc = auc(recall, precision)

 
print(pr_auc)
