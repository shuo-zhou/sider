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
from sider import SIDeRSVM


def cross_val(clf, X, D, y, X_test=None, D_test=None, test_size=0.2, n_split=10):
    sss = StratifiedShuffleSplit(n_splits=n_split, test_size=test_size,
                                 train_size=1 - test_size, random_state=144)
    acc = []
    for train, test in sss.split(X, y):
        # y_temp = np.zeros(n)
        # y_temp[train] = y[train]
        # disvm.fit(X, y_temp, D)
        X_test_ = np.concatenate((X[test], X_test))
        D_test_ = np.concatenate((D[test], D_test))
        clf.fit(X[train], y[train], D[train], X_test_, D_test_)
        y_pred = clf.predict(X[test])
        acc.append(accuracy_score(y[test], y_pred))

    return acc


def get_param(X, D, y, kernel='linear'):
    # n_comps = [20, 40, 50, 60, 80, 100]
    lambdas = np.logspace(-5, 4, 10)
    Cs = np.logspace(-5, 4, 10)
    gammas = np.logspace(-5, 4, 10)
    best_params = {'lambda': 1, 'C': 1, 'gamma': 1}
    best_acc = 0  

    # n = X.shape[0]
    if kernel == 'rbf':
        for gamma in gammas:
            clf = SIDeRSVM(C=best_params['C'], lambda_=best_params['lambda'], 
                           gamma=gamma, kernel=kernel, solver='osqp')
            acc = cross_val(clf, X, D, y)
            if best_acc < np.mean(acc):
                best_acc = np.mean(acc)
                best_params['gamma'] = gamma
            print('gamma:', gamma, 'Score:', np.mean(acc))

    for lambda_ in lambdas:
        clf = SIDeRSVM(lambda_=lambda_, C=best_params['C'], kernel=kernel, solver='osqp')
        acc = cross_val(clf, X, D, y)
        if best_acc < np.mean(acc):
            best_acc = np.mean(acc)
            best_params['lambda'] = lambda_
        print('Lambda:', lambda_, 'Score:', np.mean(acc))

    for C in Cs:
        clf = SIDeRSVM(C=C, lambda_=best_params['lambda'], 
                      kernel=kernel, solver='osqp')
        acc = cross_val(clf, X, D, y)
        if best_acc < np.mean(acc):
            best_acc = np.mean(acc)
            best_params['C'] = C
        print('C:', C, 'Score:', np.mean(acc))
        
    best_clf = SIDeRSVM(C=best_params['C'], kernel=kernel, gamma=best_params['gamma'],
                        lambda_=best_params['lambda'], solver='osqp')
    return best_params, best_clf
       
    
config = commandline()
data2load = config.data
# clf = make_pipeline(StandardScaler(),SVC(kernel = 'linear', max_iter = 10000))
# clf = SVC(kernel = 'linear', max_iter = 10000)
# clf = LogisticRegression()
# =============================================================================
# kwargs = {'vectorize': True, 'type_': 'alff'}
# kwargs = {'vectorize': config.vectorize, 'prob': config.prob, 'resize': True, 
#          'scale': 0.3}
kwargs = {'vectorize': config.vectorize, 'prob': config.prob, 'resize': False}
k_split = 2
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
D = np.concatenate(D_all, axis = 1)
    
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

print('Performing SIDeRSVM')
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
        best_params, clf = get_param(X_train, D_train, y_train, kernel='linear')
        # best_params, clf = get_param(X_train, D_train, y_train)
        print('Best param: ', best_params)
        
        clf.fit(X_train, y_train, D_train, Xt[test], Dt[test])
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
