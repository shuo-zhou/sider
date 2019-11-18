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

#base_dir = '/home/shuoz/data/openfmri/icml2019/zstats/'
#out_dir = '/home/shuoz/data/openfmri/icml2019/'
#data_dict = {'ds007': {20: ['c6', 'c7']}, 'ds008': {15: ['c8', 'c10']}, 
#            'ds101': {21: ['c7', 'c8']}, 'ds102': {26: ['c7', 'c8']}}

#label_dict = {'ds007': {'c6': 1, 'c7': -1}, 'ds008': {'c8': 1, 'c10': -1},
#              'ds101': {'c7': 1, 'c8': -1}, 'ds102': {'c7': 1, 'c8': -1}}

base_dir = '/shared/tale2/Shared/szhou/ICML2019/data/openfmri/zstats2/'
out_dir = '/shared/tale2/Shared/szhou/ICML2019/data/openfmri/'
data_dict = {'ds007': {20: ['c2', 'c3']}, 'ds008': {15: ['c3', 'c4']}, 
            'ds101': {21: ['c1', 'c2']}, 'ds102': {26: ['c1', 'c2']}}

label_dict = {'ds007': {'c2': 1, 'c3': -1}, 'ds008': {'c3': 1, 'c4': -1},
              'ds101': {'c1': 1, 'c2': -1}, 'ds102': {'c1': 1, 'c2': -1}}

def img2array(img_path, data_dict, label_dict):
    img_list = []
    img_sample = None
    n_img = len(os.listdir(img_path))
    
    label = pd.DataFrame({'Dataset': np.full(n_img, np.nan), 
                      'Sub': np.full(n_img, np.nan),
                      'Contrast': np.full(n_img, np.nan),
                      'Run': np.full(n_img, np.nan),
                      'Label': np.full(n_img, np.nan)})
    i = 0
    for ds in data_dict:
        for n_sample in data_dict[ds]:
            for sub_id in range(n_sample):
                for contrast in data_dict[ds][n_sample]:
                    for run in [1, 2]:
                        file_name = '%s_sub%03d_%s_%s.nii.gz'%(ds, sub_id+1, 
                                                               contrast, run)
                        full_path = os.path.join(img_path, file_name)
                        if os.path.isfile(full_path):
                            label.loc[i, 'Dataset'] = ds
                            label.loc[i, 'Sub'] = 'sub%03d'%(sub_id+1)
                            label.loc[i, 'Contrast'] = contrast
                            label.loc[i, 'Run'] = run
                            label.loc[i, 'Label'] = label_dict[ds][contrast]
                            i += 1
    
                            img = nib.load(full_path)
                            img_list.append(img.get_data())
            
                            if img_sample == None:
                                img_sample = img
    img_array = np.stack(img_list,axis=-1)
    
    return img_array, img_sample, label
#

img_array, img_sample, label = img2array(base_dir, data_dict, label_dict)
label.to_csv('%sopenfmri_label.csv'%out_dir, index = False)

img2save = nib.Nifti1Image(img_array, img_sample.affine)
img2save.to_filename(os.path.join(out_dir, 'openfmri_img.nii.gz'))
#np.save('%sopenfmri_array.npy'%out_dir, img_array)

img_data = img_sample.get_data()
badsubthresh = 3

maskvox=np.where(img_data>=0)
goodvox=np.zeros(img_data.shape)
missing_count=np.zeros(img_data.shape)

for v in range(len(maskvox[0])):
    x=maskvox[0][v]
    y=maskvox[1][v]
    z=maskvox[2][v]
    if not np.sum(img_array[x,y,z,:]==0.0)>badsubthresh:
        goodvox[x,y,z]=1
    missing_count[x,y,z]= np.sum(img_array[x,y,z,:]==0.0)
    
print('Creating mask image...')
new_mask=nib.Nifti1Image(goodvox, img_sample.affine)
new_mask.to_filename(os.path.join(out_dir,'goodvoxmask_openfmri.nii.gz')) 
print('Done!')
img_vec = img_array[np.where(goodvox)]  
np.save('%sopenfmri_vec.npy'%out_dir, img_vec)
