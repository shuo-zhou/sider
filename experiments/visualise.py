#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:32:19 2019

@author: shuoz
"""

import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
import nibabel as nib
from nilearn import plotting


def plot_coef(coef_img, img_name, thre_rate=0.01):
    coef = coef_img.get_data()
    coef_vec = tl.tensor_to_vec(coef)
    # selection = SelectPercentile(f_classif, percentile=thre_rate)
    n_voxel_th = int(coef_vec.shape[0] * thre_rate)
    top_voxel_idx = (abs(coef_vec)).argsort()[::-1][:n_voxel_th]
    thre = coef_vec[top_voxel_idx[-1]]
    # coef_to_plot = np.zeros(coef.shape[0])
    # coef_to_plot[top_voxel_idx] = coef[top_voxel_idx]
    # thre = np.amax(abs(coef)) * thre_rate # high absulte value times threshold rate
    # coef_img = nib.Nifti1Image(coef, maskimg.affine)
    # plotting.plot_stat_map(coef_img, threshold=thre, output_file='%s.png'%img_name, cut_coords=(0, 15, 55))
    plotting.plot_stat_map(coef_img, threshold=thre, output_file='%s_.pdf' % img_name, display_mode='x',
                           vmax=0.0004, cut_coords=range(0, 1, 1), colorbar=False)
    # plotting.plot_stat_map(coef_img, threshold=thre, output_file='%s.png' % img_name)


basedir = 'D:/icml2019/data/openfmri'
mask = os.path.join(basedir, 'goodvoxmask_openfmri.nii.gz')
# os.path.join(basedir,'goodvoxmask.nii.gz')
maskimg = nib.load(mask)
maskdata = maskimg.get_data()
maskvox = np.where(maskdata)
plt.rcParams.update({'font.size': 14})
# sample_img = nib.load('ds007_sub001_c6_1.nii.gz')

algs = ['SIDeR', 'ARTL', 'SVM']
probs = ['ABandC']
for prob in probs:
    for alg in algs:
        coef_img = nib.load(os.path.join(basedir, 'aaai_%s%s.nii.gz'%(alg, prob)))
        plot_coef(coef_img, alg)
