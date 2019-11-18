import warnings
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import pairwise_kernels


def info2idx(y):
    n_sample = y.shape[0]
    label_unique = np.unique(y)
    n_unique = label_unique.shape[0]
    idx = np.zeros(n_sample)
    for i in range(n_unique):
        idx[np.where(y == label_unique[i])] = i
    return idx


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


def get_clf(X, y, cv=None, kernel='rbf', k_split=10, test_size=0.2):
    warnings.filterwarnings("ignore", category=ConvergenceWarning,
                            module="sklearn")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    if kernel == 'linear':
        svc = SVC(kernel='linear', max_iter=10000)
        param_grid = {'C': np.logspace(-2, 5, 8)}
    else:
        svc = SVC(kernel='rbf', max_iter=10000)
        param_grid = {'C': np.logspace(-2, 5, 8), 
                      'gamma': np.logspace(-6, 2, 9)}
    if cv is None:
        cv = StratifiedShuffleSplit(n_splits=k_split, test_size=test_size, 
                                    train_size=1 - test_size, random_state=144)
    search = GridSearchCV(svc, param_grid, n_jobs=4, cv=cv, iid=False)
    search.fit(X, y)
    return search


def scale_tl(Xs, Xt):
    scaler = StandardScaler()
    X = np.concatenate((Xs, Xt))
    scaler.fit(X)
    return scaler.transform(Xs), scaler.transform(Xt)


def get_hsic(X, Y, kernel_x='linear', kernel_y='linear', **kwargs):
    n = X.shape[0]
    I = np.eye(n)
    H = I - 1. / n * np.ones((n, n))
    Kx = pairwise_kernels(X, metric=kernel_x, **kwargs)
    Ky = pairwise_kernels(Y, metric=kernel_y, **kwargs)
    return 1 / np.square(n - 1) * np.trace(np.linalg.multi_dot([Kx, H, Ky, H]))
