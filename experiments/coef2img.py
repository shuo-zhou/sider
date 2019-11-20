#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 15:28:32 2019

@author: shuoz
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 10:40:54 2019

@author: shuoz
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score#, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.feature_selection import SelectPercentile, f_classif
import matplotlib.pyplot as plt
from matplotlib import rc
from sider import SIDeRSVM
from utils.load_data import load_data
import nibabel as nib
from nilearn import plotting
from sklearn.svm import SVC


def info2onehot(y):
    n_sample = y.shape[0]
    label_unique = np.unique(y)
    n_unique = label_unique.shape[0]
    A = np.zeros((n_sample, n_unique))
    for i in range(len(label_unique)):        
        A[np.where(y == label_unique[i]), i] = 1
    return A


def cat_onehot(X1, X2):
    n_row1 = X1.shape[0]
    n_col1 = X1.shape[1]
    
    n_row2 = X2.shape[0]
    n_col2 = X2.shape[1]
    
    X = np.zeros((n_row1+n_row2, n_col1+n_col2))
    X[:n_row1, :n_col1] = X1
    X[n_row1:, n_col1:] = X2
    return X


def get_hsic(X, Y, kernel_x='linear', kernel_y='linear', **kwargs):
    n = X.shape[0]
    I = np.eye(n)
    H = I - 1. / n * np.ones((n, n))
    Kx = pairwise_kernels(X, metric=kernel_x, **kwargs)
    Ky = pairwise_kernels(Y, metric=kernel_y, **kwargs)
    return 1/np.square(n-1) * np.trace(np.linalg.multi_dot([Kx, H, Ky, H]))


def plot_coef(coef, img_name, maskimg, maskvox, thre_rate=0.05):
    coef = coef.reshape(-1)
    # selection = SelectPercentile(f_classif, percentile=thre_rate)
    n_voxel_th = int(coef.shape[0] * thre_rate)
    top_voxel_idx = (abs(coef)).argsort()[::-1][:n_voxel_th]
    thre = coef[top_voxel_idx[-1]]
#    coef_to_plot = np.zeros(coef.shape[0])
#    coef_to_plot[top_voxel_idx] = coef[top_voxel_idx]
#    thre = np.amax(abs(coef)) * thre_rate # high absulte value times threshold rate
    coef_array = np.zeros((91, 109, 91))
    coef_array[maskvox] = coef
    coef_img = nib.Nifti1Image(coef_array, maskimg.affine)
    # coef_img_file = '/home/shuoz/data/openfmri/coef_img/nips19/%s.nii.gz'%img_name
    coef_img_file = 'D:/icml2019/data/openfmri/aaai_%s.nii.gz' % img_name
    nib.save(coef_img, coef_img_file)

    # plotting.plot_glass_brain(coef_img, threshold=thre, output_file='%s.pdf'%img_name,
    #                           vmax=0.0004, plot_abs=False, display_mode='z', colorbar=False)
#    plotting.plot_glass_brain(coef_img, threshold = thre, output_file='%s.pdf'%img_name, 
#                              vmax = 5.3, plot_abs=False, display_mode='z')
#     plotting.plot_glass_brain(coef_img, threshold=thre, output_file='%s.pdf'%img_name,
#                               plot_abs=False, display_mode='z', colorbar=True)

#    plotting.plot_stat_map(coef_img, display_mode='z', threshold = thre,
#                           cut_coords=range(-0, 1, 1), output_file='%s.pdf'%img_name)
#     plotting.plot_stat_map(coef_img, display_mode='ortho', threshold=thre,
#                            output_file='%s.pdf'%img_name)
#    plotting.plot_stat_map(coef_img, display_mode='z', threshold = thre,
#                           cut_coords=range(-0, 1, 1), output_file='%s.png'%img_name)
    plotting.plot_stat_map(coef_img, threshold=thre, output_file='%s.pdf' % img_name,
                           cut_coords=(0, 5, 0), colorbar=False)
    plotting.plot_stat_map(coef_img, threshold=thre, output_file='%s.png' % img_name,
                           cut_coords=(0, 5, 0), colorbar=False)


data2load = 'openfmri'
# probs = {1:['A to D', 'D to A'], 2:['E to F','F to E'], 3:['B & C to A'],
#         4:['A & B to C'], 5:['A & C to B'], 6: ['B to A', 'A to B'],
#         7: ['C to A', 'A to C'], 8: ['C to B', 'B to C']}

# probs = {1:'AandD', 2:'EandF', 4:'ABandC',  6: 'AandB', 7: 'AandC',
#         8:'BandC', 9:'CandD', 0:'BandD'}

probs = {5: 'ABandC'}
# probs = {9: 'CandD'}
# probs = {3:'A', 5:'B', 4:'C', 1:'D'}
# probs = {3:'A', 5:'B', 4:'C'}

Cs = np.logspace(-3, 4, 8)
# lambdas = np.logspace(-2, 2, 5)
lambdas = [0, 0.01, 0.1, 1, 10, 100]
markers = ['o', 'v', 'x', 'D', 's']
# pair2plt = ['A to B', 'C to B', 'A & C to B']
pair2plt = ['A to C', 'B to C', 'A & B to C']  #, 'A to D', 'D to A']
# pair2plt = ['B to A', 'C to A', 'B & C to A']
# gammas = np.logspace(-4, 3, 8)

# mask = '/home/shuoz/data/openfmri/icml2019/goodvoxmask_openfmri.nii.gz'
mask = 'D:/icml2019/data/openfmri/goodvoxmask_openfmri.nii.gz'
# os.path.join(basedir,'goodvoxmask.nii.gz')
maskimg = nib.load(mask)
maskdata = maskimg.get_data()
maskvox = np.where(maskdata)
plt.rcParams.update({'font.size': 14})

sss = StratifiedShuffleSplit(test_size=0.5, random_state=0)
percentile = 0.1
acc = {}
for prob in probs:
    acc[prob] = {}
    kwargs = {'vectorize': True, 'prob': prob, 'resize': False}
    
    # Xs, Xt, ys_, yt_ = load_data(data=data2load, **kwargs)
    Xt, Xs, yt_, ys_ = load_data(data=data2load, **kwargs)

    X_all = np.concatenate((Xs, Xt))
    scaler = StandardScaler()
    scaler.fit(X_all)
    X_all = scaler.transform(X_all)
    Xs = scaler.transform(Xs)
    Xt = scaler.transform(Xt)
    
    y_ = pd.concat([ys_, yt_])
    y_exp = y_[['Dataset', 'Task']]
    y_exp = y_exp.drop_duplicates()
    n_exp = y_exp.shape[0]
    n_sample = y_.shape[0]
    y_ = y_.set_index(np.arange(n_sample))
    A = dict()
    A['exp'] = np.zeros((n_sample, n_exp))
    DS = list(np.unique(y_['Dataset']))
    for i in range(n_exp):
        exp = list(y_exp.iloc[i,:])
        idx = y_.loc[(y_['Dataset'].isin(exp) & y_['Task'].isin(exp))].index
        A['exp'][idx, i] = 1
        
    for ds in DS:
        idx = y_.loc[(y_['Dataset'] == ds)].index
    
        A_onehot = info2onehot(y_.loc[idx, 'Sub'])
        if 'Sub' not in A:
            A['Sub'] = A_onehot
        else:
            A['Sub'] = cat_onehot(A['Sub'], A_onehot)
    
#    A['Gender'] = info2onehot(y_['Gender'])
#    age_scaler = StandardScaler()
#    A['Age'] = age_scaler.fit_transform(y_['Age'].values.reshape((n_sample, 1)))
    
    A_all = []
    for key in A:
        A_all.append(A[key])
    D = np.concatenate(A_all, axis=1)
        
    ys = ys_['Label'].values
    ns = ys.shape[0]
    
    yt = yt_['Label'].values
    nt = yt.shape[0]

    Ds = D[:ns, :]
    Dt = D[ns:, :]
    Ds_exp = np.zeros((ns, 2))
    Ds_exp[:, 0] = 1
    Dt_exp = np.zeros((nt, 2))
    Dt_exp[:, 1] = 1
    y_all = np.concatenate((ys, yt))
    

    coefs = np.zeros((5, Xt.shape[1]))
    top_coefs = np.zeros((5, Xt.shape[1]))
    hsic_ = []
    fold = 0
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=144*10)
    for train, test in skf.split(Xt, yt):
        # SIDeR
        x_train = np.concatenate((Xs, Xt[train]))
        y_train = np.concatenate((ys, yt[train]))
        # D_train = np.concatenate((Ds, Dt[train]))
        # clf = SIDeRSVM(C=0.1, lambda_=100)
        # clf.fit(x_train, Xt[test], y_train, D_train, Dt[test])
        # # img_name = 'SIDeR%sFold%s' % (probs[prob], fold)
        # coef_ = np.dot(clf.coef_, X_all)

        # ARTL
        D_train = np.concatenate((Ds_exp, Dt_exp[train]))
        clf = SIDeRSVM(C=0.1, lambda_=100)
        clf.fit(x_train, y_train, D_train, Xt[test], Dt_exp[test])
        coef_ = np.dot(clf.coef_, X_all)

        # linear svm
        # x_train = Xt[train]
        # y_train = yt[train]
        # clf = SVC(kernel='linear')
        # clf.fit(x_train, y_train)
        # img_name = 'svm%sFold%s' % (probs[prob], fold)
        # coef_ = clf.coef_

        # n_voxel_th = int(coef_.shape[0] * percentile)
        # top_voxel_idx = (abs(coef_)).argsort()[::-1][:n_voxel_th]
        coefs[fold, :] = coef_
        # top_coefs[fold, top_voxel_idx] = coef_[top_voxel_idx]
        score = clf.decision_function(X_all).reshape(-1, 1)
        hsic_.append(get_hsic(score, D))

        fold += 1

    # for i in range(5):
    #     print(np.count_nonzero(top_coefs[i, :]))
    #     top_coefs[:, np.where(top_coefs[i, :] == 0)] = 0
    # n_overlap = np.count_nonzero(top_coefs, axis=1)

    # clf = SVC(kernel='linear')
    # clf.fit(Xt, yt)
    # coef_ = clf.coef_

    coef_mean = np.mean(coefs, axis=0)
    # img_name = 'SIDeR%s' % (probs[prob])
    img_name = 'ARTL%s' % (probs[prob])
    # img_name = 'SVM%s' % (probs[prob])
    # img_name = 'SVM_all%s' % (probs[prob])
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # coef_scal = scaler.fit_transform(coef_mean.reshape(-1, 1))
    # plot_coef(coef_mean, img_name, maskimg, maskvox, thre_rate=0.1)

    coef_array = np.zeros((91, 109, 91))
    coef_array[maskvox] = coef_mean
    coef_img = nib.Nifti1Image(coef_array, maskimg.affine)
    # coef_img_file = '/home/shuoz/data/openfmri/coef_img/nips19/%s.nii.gz'%img_name
    coef_img_file = 'D:/icml2019/data/openfmri/aaai_%s.nii.gz' % img_name
    nib.save(coef_img, coef_img_file)
