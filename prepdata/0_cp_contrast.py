#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:51:30 2018

@author: shuoz
"""


import os

BASEDIR = '/shared/tale2/Shared/szhou/openfmri/preprocessed/'
OUTFOLDER = '/shared/tale2/Shared/szhou/data/openfmri/'
#CONTRAST_DICT = {'ds007':{'task003': [6, 7]},
#                'ds008':{'task002': [8, 10]},
#                'ds101':{'task001': [7, 8]},
#                'ds102':{'task001': [7, 8]}}
CONDITION_DICT = {'ds007':{'task001': [2, 3], 'task002': [2, 3], 'task003': [2, 3]},
                  'ds008':{'task001': [2, 3], 'task002': [3, 4]},
                  'ds101':{'task001': [1, 3]}, 'ds102':{'task001': [1, 3]}}



f = open('copy.sh', 'w')

for ds in CONDITION_DICT:
    ds_folder = '%s%s'%(BASEDIR, ds)
    for sub_folder in os.listdir(ds_folder):
        if sub_folder.startswith('sub'):
            for task in CONDITION_DICT[ds]:
                for run in [1,2]:
                    feat_folder = '%s/%s/model/model001/%s_run00%s.feat/'%(ds_folder,sub_folder,task,run)
                    if os.path.exists(feat_folder):
                        for cond in CONDITION_DICT[ds][task]:
                            img_target = '%szstats2/%s_%s_%s_c%s_%s.nii.gz'%(OUTFOLDER, ds, task, sub_folder, cond, run)
                            img_path = '%sreg_standard/stats/zstat%s.nii.gz'%(feat_folder, cond)
                            f.write('cp %s %s\n'%(img_path, img_target))
                    
f.close()
