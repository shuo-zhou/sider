#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 14:52:16 2018

@author: shuoz
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from utils.load_data import load_data
from utils.cmdline import commandline
from utils.funcs import info2onehot, cat_onehot, info2idx
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from sider import SIDeRSVM, SIDeRLS


def cross_val(clf, X, y, D, test_size=0.2, n_split=10):
    sss = StratifiedShuffleSplit(n_splits=n_split, test_size=test_size,
                                 train_size=1 - test_size, random_state=144)
    acc = []
    for train, test in sss.split(X, y):
        # y_temp = np.zeros(n)
        # y_temp[train] = y[train]
        # disvm.fit(X, y_temp, D)
        # X_test_ = np.concatenate((X[test], X_test))
        # D_test_ = np.concatenate((D[test], D_test))
        # clf.fit(X[train], y[train], D[train], X_test_, D_test_)
        clf.fit(X, y[train], D)
        y_pred = clf.predict(X[test])
        acc.append(accuracy_score(y[test], y_pred))

    return acc


def search_param(X, D, y, kernel='linear', loss='hinge'):
    param_grids = {'ls': {'linear': {'lambda_': np.logspace(-4, 3, 8),
                                     'sigma_': np.logspace(-4, 3, 8)},
                          'rbf': {'gamma': np.logspace(-4, 2, 7),
                                  'sigma_': np.logspace(-4, 3, 8),
                                  'lambda_': np.logspace(-4, 3, 8)}},
                   'hinge': {'linear': {'C': np.logspace(-4, 3, 8),
                                        'lambda_': np.logspace(-4, 3, 8)},
                             'rbf': {'gamma': np.logspace(-4, 2, 7),
                                     'C': np.logspace(-4, 3, 8),
                                     'lambda_': np.logspace(-4, 3, 8)}}}
    algs = {'ls': SIDeRLS, 'hinge': SIDeRSVM}
    default_params = {'hinge': {'C': 1, 'lambda_': 1, 'gamma': 0.001,
                                'kernel': kernel, 'solver': 'osqp'},
                      'ls': {'lambda_': 1, 'sigma_': 1, 'gamma': 0.001,
                             'kernel': kernel}}
    param_grid = param_grids[loss][kernel]
    alg = algs[loss]
    best_params = default_params[loss].copy()

    # best_acc = 0

    for param in param_grid:
        kwd_params = {key: best_params[key] for key in best_params if key != param}
        best_acc = 0
        for param_val in param_grid[param]:
            kwd_ = kwd_params.copy()
            kwd_[param] = param_val
            clf = alg(**kwd_)
            acc = cross_val(clf, X, y, D)
            acc_avg = np.mean(acc)
            if best_acc < acc_avg:
                best_acc = acc_avg
                best_params[param] = param_val
            print(kwd_, 'Score:', acc_avg)

        best_clf = alg(**best_params)

    return best_params, best_clf

    
config = commandline()
data2load = config.data
loss = config.loss
krnl = config.kernel
# clf = make_pipeline(StandardScaler(),SVC(kernel = 'linear', max_iter = 10000))
# clf = SVC(kernel = 'linear', max_iter = 10000)
# clf = LogisticRegression()
# =============================================================================
# kwargs = {'vectorize': True, 'type_': 'alff'}
# kwargs = {'vectorize': config.vectorize, 'prob': config.prob, 'resize': True, 
#          'scale': 0.3}
kwargs = {'vectorize': True, 'prob': config.prob, 'resize': False}
k_split = config.kfold
print('K-fold:', k_split)

if config.swap:
    Xt, Xs, yt_, ys_ = load_data(data=data2load, **kwargs)
else:
    Xs, Xt, ys_, yt_ = load_data(data=data2load, **kwargs)
# X0 = unfold(X0_, -1)
# X1 = unfold(X1_, -1)

y_ = pd.concat([ys_, yt_])
y_exp = y_[['Dataset', 'Task']]
y_exp = y_exp.drop_duplicates()
n_exp = y_exp.shape[0]
n_sample = y_.shape[0]
y_ = y_.set_index(np.arange(n_sample))
D = {}
DS = list(np.unique(y_['Dataset']))
D['exp'] = np.zeros((n_sample, n_exp))
for i in range(n_exp):
    exp = list(y_exp.iloc[i,:])
    idx = y_.loc[(y_['Dataset'].isin(exp) & y_['Task'].isin(exp))].index
    D['exp'][idx, i] = 1
    
for ds in DS:
    idx = y_.loc[(y_['Dataset'] == ds)].index

    D_onehot = info2onehot(y_.loc[idx, 'Sub'])
    if 'Sub' not in D:
        D['Sub'] = D_onehot
    else:
        D['Sub'] = cat_onehot(D['Sub'], D_onehot)

# D['Gender'] = info2onehot(y_['Gender'])
# age_scaler = StandardScaler()
# D['Age'] = age_scaler.fit_transform(y_['Age'].values.reshape((n_sample, 1)))

D_all = []
for key in D:
    D_all.append(D[key])
    print(key)
D = np.concatenate(D_all, axis=1)
    
ys = ys_['Label'].values
ns = ys.shape[0]

yt = yt_['Label'].values
nt = yt.shape[0]
tsub_idx = info2idx(yt_.loc[:, 'Sub'])
nt_sub = np.unique(tsub_idx).shape[0]

# clf1 = make_pipeline(PCA(n_components = 45), SVC(kernel='linear'))
# clf1.fit(Xt[train], yt[train])
X_all = np.concatenate((Xs, Xt))
scaler = StandardScaler()
scaler.fit(X_all)
X_all = scaler.transform(X_all)
Xs = scaler.transform(Xs)
Xt = scaler.transform(Xt)

print('Performing SIDeR')
acc = []
auc = []
pred_all = []
Ds = D[:ns, :]
Dt = D[ns:, :]

# test_size = 0.2
# print('Test size: ', test_size)
# sss = StratifiedShuffleSplit(n_splits=20, test_size=test_size, 
#                              train_size=1-test_size, random_state=144)
# for train, test in sss.split(Xt, yt):

for i in range(10):    
    pred = np.zeros(yt.shape)
    score = np.zeros(yt.shape)
    # skf = StratifiedKFold(n_splits=k_split, shuffle=True, random_state=144 * i)
    # for train, test in skf.split(Xt, yt):
    kf = KFold(n_splits=k_split, shuffle=True, random_state=144 * i)
    for train_idx, test_idx in kf.split(np.arange(nt_sub)):
        train = np.isin(tsub_idx, train_idx)
        test = np.isin(tsub_idx, test_idx)
        D_train = np.concatenate((Ds, Dt[train]))
        X_train = np.concatenate((Xs, Xt[train]))
        y_train = np.concatenate((ys, yt[train]))
        best_params, clf = search_param(X_train, D_train, y_train, kernel=krnl)
        # best_params, clf = get_param(X_train, D_train, y_train)
        print('Best param: ', best_params)
        X = np.concatenate((X_train, X[test]))
        D_ = np.concatenate((D_train, Dt[test]))
        clf.fit(X, y_train, D_)
        pred[test] = clf.predict(Xt[test])
        score[test] = clf.decision_function(Xt[test])
        # print('Intercept: ', clf.intercept_)
# =============================================================================        
    # pred_all.append(pred)
    # print('True label: ', yt[test])
    # print('Prediction: ', pred)
    # acc.append(accuracy_score(yt[test], pred))

    print('True label: ', yt)
    print('Prediction: ', pred)
    print('Score: ', score)
    acc.append(accuracy_score(yt, pred))
    auc.append(roc_auc_score(yt, score))
        
print('Acc Mean:', np.mean(acc), 'Acc std:', np.std(acc))
print(acc)

print('AUC Mean:', np.mean(auc), 'AUC std:', np.std(auc))
print(auc)
