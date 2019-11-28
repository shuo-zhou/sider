#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 17:04:47 2018

@author: shuoz
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib

#base_dir = '/home/shuoz/data/openfmri/zstats/'
#out_dir = '/home/shuoz/data/openfmri/'
#data_dict = {'ds007': {20: ['c6', 'c7']}, 'ds008': {15: ['c8', 'c10']}, 
#            'ds101': {21: ['c7', 'c8']}, 'ds102': {26: ['c7', 'c8']}}

#label_dict = {'ds007': {'c6': 1, 'c7': -1}, 'ds008': {'c8': 1, 'c10': -1},
#              'ds101': {'c7': 1, 'c8': -1}, 'ds102': {'c7': 1, 'c8': -1}}

base_dir = '/shared/tale2/Shared/szhou/data/openfmri/zstats2/'
out_dir = '/shared/tale2/Shared/szhou/data/openfmri/'
data_dict = {'ds007': 20, 'ds008': 15, 'ds101': 21, 'ds102': 26}

#label_dict = {'ds007': {'c2': 1, 'c3': -1}, 'ds008': {'c3': 1, 'c4': -1},
#              'ds007': {'c2': 1, 'c3': -1}, 'ds008': {'c3': 1, 'c4': -1},
#              'ds101': {'c1': 1, 'c2': -1}, 'ds102': {'c1': 1, 'c2': -1}}

label_dict = {'ds007': {'task001': ['c2', 'c3'], 'task002': ['c2', 'c3'], 
              'task003': ['c2', 'c3']}, 'ds008': {'task001': ['c2', 'c3'],
              'task002': ['c3', 'c4']}, 'ds101': {'task001': ['c1', 'c3']},
              'ds102': {'task001': ['c1', 'c3']}}

def img2array(img_path, data_dict, label_dict):
    img_list = []
    img_sample = None
    n_img = len(os.listdir(img_path))
    
    label = pd.DataFrame({'Dataset': np.full(n_img, np.nan), 
                          'Task': np.full(n_img, np.nan),
                          'Sub': np.full(n_img, np.nan),
                          'Condition': np.full(n_img, np.nan),
                          'Run': np.full(n_img, np.nan),
                          'Label': np.full(n_img, np.nan)})
    i = 0
    for ds in data_dict:
        for task in label_dict[ds]:
            n_sample = data_dict[ds]
            for sub_id in range(n_sample):
                for condition in label_dict[ds][task]:
                    y = label_dict[ds][task].index(condition)
                    if y == 0:
                        y = 1
                    else:
                        y = -1
                    for run in [1, 2]:
                        file_name = '%s_%s_sub%03d_%s_%s.nii.gz'%(ds, task, sub_id+1, 
                                                                condition, run)
                        full_path = os.path.join(img_path, file_name)
                        if os.path.isfile(full_path):
                            label.loc[i, 'Dataset'] = ds
                            label.loc[i, 'Task'] = task
                            label.loc[i, 'Sub'] = 'sub%03d'%(sub_id+1)
                            label.loc[i, 'Condition'] = condition
                            label.loc[i, 'Run'] = run
                            label.loc[i, 'Label'] = y
                            i += 1
        
                            img = nib.load(full_path)
                            img_list.append(img.get_data())
                
                            if img_sample == None:
                                img_sample = img
    img_array = np.stack(img_list,axis=-1)
    
    return img_array, img_sample, label

img_array, img_sample, label = img2array(base_dir, data_dict, label_dict)
# label.to_csv('%sopenfmri_label.csv'%out_dir, index = False)

# img2save = nib.Nifti1Image(img_array, img_sample.affine)
# img2save.to_filename(os.path.join(out_dir, 'openfmri_img.nii.gz'))
#np.save('%sopenfmri_array.npy'%out_dir, img_array)

img_data = img_sample.get_data()
badthresh = 10

# maskvox=np.where(img_data>=0)
goodvox=np.zeros(img_data.shape)
print(goodvox.shape)
missing_count=np.zeros(img_data.shape)

# for v in range(len(maskvox[0])):
for x in range(goodvox.shape[0]):
    for y in range(goodvox.shape[1]):
        for z in range(goodvox.shape[2]):
    # x=maskvox[0][v]
    # y=maskvox[1][v]
    # z=maskvox[2][v]
            n_zeros = np.where(img_array[x,y,z,:]==0.0)[0].shape[0]
            # print(x, y, z, non_zeros)

            if n_zeros < badthresh:
                goodvox[x,y,z]=1
            missing_count[x,y,z]= np.sum(img_array[x,y,z,:]==0.0)
    
print('Creating mask image...')
new_mask=nib.Nifti1Image(goodvox, img_sample.affine)
new_mask.to_filename(os.path.join(out_dir,'goodvoxmask_openfmri.nii.gz')) 
print('Done!')
img_vec = img_array[np.where(goodvox==1)]  
np.save('%sopenfmri_vec.npy'%out_dir, img_vec)
