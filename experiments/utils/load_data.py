#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:40:43 2018

@author: shuoz
"""
import os
import sys
from scipy.io import loadmat
import numpy as np
import pandas as pd
import nibabel as nib

# path on hpc cluster
# openfmri_data_path = '/shared/tale2/Shared/szhou/ICML2019/data/openfmri/'

# path on desktop
openfmri_data_path = '/home/shuoz/data/openfmri/icml2019/'

# path on laptop
# openfmri_data_path = 'D:/icml2019/data/openfmri/'


def load_openfmri_data(vectorize=True, prob=1, resize=False, scale=None):
    probs = {1: {0: ['ds007', 'task003', 'c2', 'c3'],
                 1: ['ds008', 'task002', 'c3', 'c4']},
             2: {0: ['ds101', 'task001', 'c1', 'c3'],
                 1: ['ds102', 'task001', 'c1', 'c3']},
             3: {0: ['ds007', 'task001', 'task002', 'c2', 'c3'],
                 1: ['ds007', 'task003', 'c2', 'c3']},
             4: {0: ['ds007', 'task002', 'task003', 'c2', 'c3'],
                 1: ['ds007', 'task001', 'c2', 'c3']},
             5: {0: ['ds007', 'task001', 'task003', 'c2', 'c3'],
                 1: ['ds007', 'task002', 'c2', 'c3']},
             6: {0: ['ds007', 'task002', 'c2', 'c3'],
                 1: ['ds007', 'task003', 'c2', 'c3']},
             7: {0: ['ds007', 'task001', 'c2', 'c3'],
                 1: ['ds007', 'task003', 'c2', 'c3']},
             8: {0: ['ds007', 'task001', 'c2', 'c3'],
                 1: ['ds007', 'task002', 'c2', 'c3']},
             9: {0: ['ds007', 'task001', 'c2', 'c3'],
                 1: ['ds008', 'task002', 'c3', 'c4']},
             0: {0: ['ds007', 'task002', 'c2', 'c3'],
                 1: ['ds008', 'task002', 'c3', 'c4']}}

    if resize and scale is not None:
        X0_mat = loadmat(os.path.join(openfmri_data_path, '%s%s_%02d.mat'
                                      % (probs[prob][0][0], probs[prob][0][1], scale * 10)))
        X0 = X0_mat['X'].astype(np.float)
        y0 = X0_mat['y'].astype(np.float)[0, :]

        X1_mat = loadmat(os.path.join(openfmri_data_path, '%s%s_%02d.mat'
                                      % (probs[prob][1][0], probs[prob][1][1], scale * 10)))
        X1 = X1_mat['X'].astype(np.float)
        y1 = X1_mat['y'].astype(np.float)[0, :]

    else:
        label = pd.read_csv(os.path.join(openfmri_data_path, 'openfmri_label.csv'))
        idx = dict()
        idx[0] = label.loc[(label['Dataset'].isin(probs[prob][0])) &
                           (label['Task'].isin(probs[prob][0])) &
                           (label['Condition'].isin(probs[prob][0][2:]))].index
        idx[1] = label.loc[(label['Dataset'].isin(probs[prob][1])) &
                           (label['Task'].isin(probs[prob][1])) &
                           (label['Condition'].isin(probs[prob][1][2:]))].index

        y0 = label.loc[idx[0]]
        y1 = label.loc[idx[1]]

    if vectorize:
        X = np.load(os.path.join(openfmri_data_path, 'openfmri_vec.npy'))
        X = X.T
        X0 = X[idx[0], :]
        X1 = X[idx[1], :]
    else:
        X = nib.load(os.path.join(openfmri_data_path, 'openfmri_img.nii.gz'
                                  )).get_data()
        X0 = X[:, :, :, idx[0]]
        X1 = X[:, :, :, idx[1]]

    return X0, X1, y0, y1


def load_data(data='openfmri', **kwargs):
    if data == 'openfmri':
        return load_openfmri_data(**kwargs)
    else:
        print('Invalid dataset')
        sys.exit()
