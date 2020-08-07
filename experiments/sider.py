#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 11:15:26 2019
Ref: Zhou, S., Li, W., Cox, C.R. and Lu, H., 2020. Side Information Dependence
 as a Regulariser for Analyzing Human Brain Conditions across Cognitive Experiments.
 In Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI 2020).
"""

import sys
import warnings
import numpy as np
from scipy.linalg import sqrtm
import scipy.sparse as sparse
from numpy.linalg import multi_dot, inv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelBinarizer
# import cvxpy as cvx
# from cvxpy.error import SolverError
from cvxopt import matrix, solvers
import osqp


def lapnorm(X, n_neighbour=3, metric='cosine', mode='distance',
            normalise=True):
    """Construct Laplacian matrix

    Args:
        X (array-like): input data, shape (n_samples, n_features)
        n_neighbour (int, optional): number of nearest neighbours. Defaults to 3.
        metric (str, optional): knn metric. Defaults to 'cosine'.
        mode (str, optional): knn mode. Defaults to 'distance'.
        normalise (bool, optional): If true, generate normalised Laplacian matrix. Defaults to True.

    Returns:
        [array-like]: Laplacian matrix, shape (n_samples, n_samples)
    """
    n = X.shape[0]
    knn_graph = kneighbors_graph(X, n_neighbour, metric=metric,
                                 mode=mode).toarray()
    W = np.zeros((n, n))
    knn_idx = np.logical_or(knn_graph, knn_graph.T)
    if mode == 'distance':
        graph_kernel = pairwise_distances(X, metric=metric)
        W[knn_idx] = graph_kernel[knn_idx]
    else:
        W[knn_idx] = 1

    D = np.diag(np.sum(W, axis=1))
    if normalise:
        D_ = inv(sqrtm(D))
        lapmat = np.eye(n) - multi_dot([D_, W, D_])
    else:
        lapmat = D - W
    return lapmat


def solve_semi_dual(K, y, Q_, C, solver='osqp'):
    if y.ndim > 2:
        print('Invalid shape of y')
        sys.exit()
    n_class = y.shape[1]
    coef_list = []
    support_ = []
    for i in range(n_class):
        coef, support = semi_binary_dual(K, y[:, i], Q_, C, solver)
        coef_list.append(coef.reshape(-1, 1))
        support_.append(support)

    coef_ = np.concatenate(coef_list, axis=1)

    return coef_, support_


def semi_binary_dual(K, y_, Q_, C, solver='osqp'):
    """
    Construct & solve quraprog problem
    :param K:
    :param y_:
    :param Q_:
    :param C:
    :param solver:
    :return:
    """
    nl = y_.shape[0]
    n = K.shape[0]
    J = np.zeros((nl, n))
    J[:nl, :nl] = np.eye(nl)
    Q_inv = inv(Q_)
    Y = np.diag(y_.reshape(-1))
    Q = multi_dot([Y, J, K, Q_inv, J.T, Y])
    Q = Q.astype('float32')
    alpha = _quadprog(Q, y_, C, solver)
    coef_ = multi_dot([Q_inv, J.T, Y, alpha])
    support_ = np.where((alpha > 0) & (alpha < C))
    return coef_, support_


def _quadprog(Q, y, C, solver='osqp'):
    """
    solve quadratic programming problem
    :param y: Label, array-like, shape (nl_samples, )
    :param Q: Quad matrix, array-like, shape (n_samples, n_samples)
    :return: coefficients alpha
    """
    # dual
    nl = y.shape[0]
    q = -1 * np.ones((nl, 1))

    if solver == 'cvxopt':
        G = np.zeros((2 * nl, nl))
        G[:nl, :] = -1 * np.eye(nl)
        G[nl:, :] = np.eye(nl)
        h = np.zeros((2 * nl, 1))
        h[nl:, :] = C / nl

        # convert numpy matrix to cvxopt matrix
        P = matrix(Q)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        A = matrix(y.reshape(1, -1).astype('float64'))
        b = matrix(np.zeros(1).astype('float64'))

        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)

        alpha = np.array(sol['x']).reshape(nl)

    elif solver == 'osqp':
        warnings.simplefilter('ignore', sparse.SparseEfficiencyWarning)
        P = sparse.csc_matrix((nl, nl))
        P[:nl, :nl] = Q[:nl, :nl]
        G = sparse.vstack([sparse.eye(nl), y.reshape(1, -1)]).tocsc()
        l = np.zeros((nl + 1, 1))
        u = np.zeros(l.shape)
        u[:nl, 0] = C

        prob = osqp.OSQP()
        prob.setup(P, q, G, l, u, verbose=False)
        res = prob.solve()
        alpha = res.x

    else:
        print('Invalid QP solver')
        sys.exit()

    return alpha


def solve_semi_ls(Q, y):
    n = Q.shape[0]
    nl = y.shape[0]
    Q_inv = inv(Q)
    # if len(y.shape) == 1:
    #     y_ = np.zeros(n)
    #     y_[:nl] = y[:]
    # else:
    y_ = np.zeros((n, y.shape[1]))
    y_[:nl, :] = y[:, :]
    return np.dot(Q_inv, y_)


def score2pred(scores):
    """
    Converting decision scores (probability) to predictions
    Parameter:
        scores: score matrix, array-like, shape (n_samples, n_class)
    Return:
        prediction matrix (1, -1), array-like, shape (n_samples, n_class)
    """
    n = scores.shape[0]
    y_pred_ = -1 * np.ones((n, n))
    dec_sort = np.argsort(scores, axis=1)[:, ::-1]
    for i in range(n):
        label_idx = dec_sort[i, 0]
        y_pred_[i, label_idx] = 1

    return y_pred_


class SIDeRSVM(BaseEstimator, TransformerMixin):
    def __init__(self, C=1, kernel='linear', lambda_=1, mu=0, solver='osqp',
                 manifold_metric='cosine', k_neighbour=3, knn_mode='distance', **kwargs):
        """
        Parameters
            C: param for importance of slack variable
            kernel: 'rbf' | 'linear' | 'poly' (default is 'linear')
            **kwargs: kernel param
            lambda_: param for side information dependence regularisation
            mu: param for manifold regularisation (default 0, not apply)
            manifold_metric: metric for manifold regularisation
            k: number of nearest numbers for manifold regularisation
            knn_mode: default distance
            solver: quadratic programming solver, cvxopt, osqp (default)
        """
        self.kwargs = kwargs
        self.kernel = kernel
        self.lambda_ = lambda_
        self.mu = mu
        self.C = C
        self.solver = solver
        # self.scaler = StandardScaler()
        # self.coef_ = None
        # self.X = None
        # self.y = None
        # self.support_ = None
        # self.support_vectors_ = None
        # self.n_support_ = None
        self.manifold_metric = manifold_metric
        self.k_neighbour = k_neighbour
        self.knn_mode = knn_mode
        self._lb = LabelBinarizer(pos_label=1, neg_label=-1)

    def fit(self, X, y, D):
        """
        solve min_x x^TPx + q^Tx, s.t. Gx<=h, Ax=b
        Parameters:
            X: Input data, array-like, shape (n_samples, n_feautres)
            y: Label, array-like, shape (nl_samples, ) where nl_samples <= n_samples
            D: Domain covariate matrix for input data, array-like, shape (n_samples, n_covariates)
        Return:
            self
        """
        # X = self.scaler.fit_transform(X)
        n = X.shape[0]
        nl = y.shape[0]
        Kd = np.dot(D, D.T)
        K = pairwise_kernels(X, metric=self.kernel, filter_params=True, **self.kwargs)
        K[np.isnan(K)] = 0

        y_ = self._lb.fit_transform(y)

        I = np.eye(n)
        H = I - 1. / n * np.ones((n, n))
        if self.mu != 0:
            lap_norm = lapnorm(X, n_neighbour=self.k_neighbour, mode=self.knn_mode,
                               metric=self.manifold_metric)
            Q_ = I + np.dot(self.lambda_ / np.square(n - 1) * multi_dot([H, Kd, H])
                            + self.mu / np.square(n) * lap_norm, K)
        else:
            Q_ = I + self.lambda_ / np.square(n - 1) * multi_dot([H, Kd, H, K])


        if self._lb.y_type_ == 'binary':
            self.coef_, self.support_ = semi_binary_dual(K, y_, Q_, self.C,
                                                         self.solver)
            self.support_vectors_ = X[:nl, :][self.support_]
            self.n_support_ = self.support_vectors_.shape[0]

        else:
            coef_list = []
            self.support_ = []
            self.support_vectors_ = []
            self.n_support_ = []
            for i in range(y_.shape[1]):
                coef_, support_ = semi_binary_dual(K, y_[: i], Q_, self.C,
                                                   self.solver)
                coef_list.append(coef_.reshape(-1, 1))
                self.support_.append(support_)
                self.support_vectors_.append(X[:nl, :][support_][-1])
                self.n_support_.append(self.support_vectors_[-1].shape[0])
            self.coef_ = np.concatenate(coef_list, axis=1)

        # K_train = get_kernel(X_train, X, kernel=self.kernel, **self.kwargs)
        # self.intercept_ = np.mean(y[self.support_] - y[self.support_] *
        #                           np.dot(K_train[self.support_], self.coef_))/self.n_support_

        # =============================================================================
        #         beta = cvx.Variable(shape = (2 * n, 1))
        #         objective = cvx.Minimize(cvx.quad_form(beta, P) + q.T * beta)
        #         constraints = [G * beta <= h]
        #         prob = cvx.Problem(objective, constraints)
        #         try:
        #             prob.solve()
        #         except SolverError:
        #             prob.solve(solver = 'SCS')
        #
        #         self.coef_ = beta.value[:n]
        # =============================================================================

        #        a = np.dot(W + self.gamma * multi_dot([H, Ka, H]), self.lambda_*I)
        #        b = np.dot(y, W)
        #        beta = solve(a, b)

        self.X = X
        self.y = y

        return self

    def decision_function(self, X):
        """
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            prediction scores, array-like, shape (n_samples)
        """
        # check_is_fitted(self, 'X')
        # check_is_fitted(self, 'y')
        # K = get_kernel(self.scaler.transform(X), self.X,
        #                kernel=self.kernel, **self.kwargs)
        K = pairwise_kernels(X, self.X, metric=self.kernel, filter_params=True, **self.kwargs)
        return np.dot(K, self.coef_)  # +self.intercept_

    def predict(self, X):
        """
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            predicted labels, array-like, shape (n_samples)
        """
        dec = self.decision_function(X)
        if self._lb.y_type_ == 'binary':
            y_pred_ = np.sign(dec).reshape(-1, 1)
        else:
            y_pred_ = score2pred(dec)

        return self._lb.inverse_transform(y_pred_)

    def fit_predict(self, X, y, D):
        """
        solve min_x x^TPx + q^Tx, s.t. Gx<=h, Ax=b
        Parameters:
            X: Input data, array-like, shape (n_samples, n_feautres)
            y: Label, array-like, shape (nl_samples, ) where nl_samples <= n_samples
            D: Domain covariate matrix for input data, array-like, shape (n_samples, n_covariates)
        Return:
            predicted labels, array-like, shape (n_test_samples,)
        """
        self.fit(X, y, D)
        return self.predict(X)


class SIDeRLS(BaseEstimator, TransformerMixin):
    def __init__(self, sigma_=1, lambda_=1, mu=0, kernel='linear', k=3,
                 knn_mode='distance', manifold_metric='cosine',
                 class_weight=None, **kwargs):
        """
        Parameters:
            sigma_: param for model complexity (l2 norm)
            lambda_: param for side information dependence regularisation
            mu: param for manifold regularisation (default 0, not apply)
            kernel: 'rbf' | 'linear' | 'poly' (default is 'linear')
            **kwargs: kernel param
            manifold_metric: metric for manifold regularisation
            k: number of nearest numbers for manifold regularisation
            knn_mode: default distance
            class_weight: None | balance (default None)
        """
        self.kwargs = kwargs
        self.kernel = kernel
        self.sigma_ = sigma_
        self.lambda_ = lambda_
        self.mu = mu
        self.classes = None
        self.coef_ = None
        self.X = None
        self.y = None
        self.manifold_metric = manifold_metric
        self.k = k
        self.knn_mode = knn_mode
        self.class_weight = class_weight
        self._lb = LabelBinarizer(pos_label=1, neg_label=-1)

    def fit(self, X, y, D):
        """
        Parameters:
            X: Input data, array-like, shape (n_samples, n_feautres)
            y: Label, array-like, shape (nl_samples, ) where nl_samples <= n_samples
            D: Domain covariate matrix for input data, array-like, shape (n_samples, n_covariates)
        Return:
            self
        """
        n = X.shape[0]
        nl = y.shape[0]
        Kd = np.dot(D, D.T)
        K = pairwise_kernels(X, metric=self.kernel, filter_params=True, **self.kwargs)
        K[np.isnan(K)] = 0

        J = np.zeros((n, n))
        J[:nl, :nl] = np.eye(nl)

        I = np.eye(n)
        H = I - 1. / n * np.ones((n, n))

        if self.mu != 0:
            lap_norm = lapnorm(X, n_neighbour=self.k, mode=self.knn_mode,
                               metric=self.manifold_metric)
            Q_ = self.sigma_ * I + np.dot(J + self.lambda_ * nl / np.square(n - 1)
                                          * multi_dot([H, Kd, H])
                                          + self.mu * nl / np.square(n) * lap_norm, K)
        else:
            Q_ = self.sigma_ * I + np.dot(J + self.lambda_ * nl / np.square(n - 1)
                                          * multi_dot([H, Kd, H]), K)

        y_ = self._lb.fit_transform(y)
        self.coef_ = solve_semi_ls(Q_, y_)

        self.X = X
        self.y = y

        return self

    def decision_function(self, X):
        """
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            prediction scores, array-like, shape (n_samples)
        """
        # check_is_fitted(self, 'X')
        # check_is_fitted(self, 'y')
        # K = get_kernel(self.scaler.transform(X), self.X,
        #                kernel=self.kernel, **self.kwargs)
        K = pairwise_kernels(X, self.X, metric=self.kernel, filter_params=True, **self.kwargs)
        return np.dot(K, self.coef_)  # +self.intercept_

    def predict(self, X):
        """
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            predicted labels, array-like, shape (n_samples)
        """
        dec = self.decision_function(X)
        if self._lb.y_type_ == 'binary':
            y_pred_ = np.sign(dec).reshape(-1, 1)
        else:
            y_pred_ = score2pred(dec)

        return self._lb.inverse_transform(y_pred_)


    def fit_predict(self, X, y, D):
        """Fit the model and making predictions according to the given data

        Args:
            X (array-like): Input data, shape (n_samples, n_feautres)
            y (array-like): Label, shape (nl_samples, ) where nl_samples <= n_samples
            D (array-like): Domain covariate matrix for input data, shape (n_samples, n_covariates)

        Returns:
            array-like: predicted labels, shape (n_samples,)
        """
        
        self.fit(X, y, D)
        return self.predict(X)
