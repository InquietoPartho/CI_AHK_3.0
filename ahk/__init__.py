from .ci_ahk import CI_AHK_SVC
# ==========================================
# CI-AHK: Class-Imbalance–Aware Hybrid Kernel SVM
# ==========================================

import numpy as np
from typing import Callable, List, Optional, Dict, Union

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import euclidean_distances, polynomial_kernel, linear_kernel
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


# ==========================================
# Kernel primitives
# ==========================================
def rbf_kernel(X, Y, gamma):
    return np.exp(-gamma * euclidean_distances(X, Y, squared=True))

def poly_kernel(X, Y, degree, coef0):
    return polynomial_kernel(X, Y, degree=degree, coef0=coef0)

def lin_kernel(X, Y):
    return linear_kernel(X, Y)


# ==========================================
# Kernel utilities
# ==========================================
def ensure_psd(K, eps=1e-10):
    K = 0.5 * (K + K.T)
    w, V = np.linalg.eigh(K)
    w = np.clip(w, eps, None)
    return (V * w) @ V.T


def project_simplex(v):
    v = np.asarray(v)
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)


# ==========================================
# CI-AHK weight learning
# ==========================================
def class_conditional_alignment_weights(Ks, y):
    """
    Learn separate kernel weights for positive and negative classes
    using class-conditional kernel alignment.
    """
    y = np.asarray(y)

    # Ensure labels in {-1, +1}
    if set(np.unique(y)) != {-1, 1}:
        y = np.where(y == y.max(), 1, -1)

    def _align(mask):
        Ky = np.outer(mask, mask)
        Ky /= (np.linalg.norm(Ky, "fro") + 1e-12)

        scores = []
        for K in Ks:
            Km = K / (np.linalg.norm(K, "fro") + 1e-12)
            scores.append(np.sum(Km * Ky))

        s = np.maximum(scores, 0)
        if s.sum() == 0:
            return np.ones(len(Ks)) / len(Ks)
        return project_simplex(np.array(s))

    pos_mask = (y == 1).astype(float)
    neg_mask = (y == -1).astype(float)

    lambda_pos = _align(pos_mask)
    lambda_neg = _align(neg_mask)

    return lambda_pos, lambda_neg


# ==========================================
# CI-AHK kernel mixing
# ==========================================
def mix_class_aware_kernels(Ks, lambda_pos, lambda_neg, y):
    """
    Construct class-aware hybrid kernel matrix.
    """
    n = y.shape[0]
    Kmix = np.zeros_like(Ks[0], dtype=float)

    lambda_mix = 0.5 * (lambda_pos + lambda_neg)

    for m, K in enumerate(Ks):
        for i in range(n):
            for j in range(n):
                if y[i] == y[j] == 1:
                    Kmix[i, j] += lambda_pos[m] * K[i, j]
                elif y[i] == y[j] == -1:
                    Kmix[i, j] += lambda_neg[m] * K[i, j]
                else:
                    Kmix[i, j] += lambda_mix[m] * K[i, j]

    return ensure_psd(Kmix)


# ==========================================
# CI-AHK SVM Classifier
# ==========================================
class CI_AHK_SVC(BaseEstimator, ClassifierMixin):
    """
    Class-Imbalance–Aware Adaptive Hybrid Kernel SVM
    """

    def __init__(
        self,
        C=1.0,
        gamma=0.1,
        degree=3,
        coef0=1.0,
        use_rbf=True,
        use_poly=True,
        use_linear=True,
        probability=False,
        class_weight=None,
        standardize=True,
        random_state=None,
    ):
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

        self.use_rbf = use_rbf
        self.use_poly = use_poly
        self.use_linear = use_linear

        self.probability = probability
        self.class_weight = class_weight
        self.standardize = standardize
        self.random_state = random_state

    # --------------------------------------
    # Kernel stack
    # --------------------------------------
    def _build_kernels(self, X, Y):
        Ks = []
        names = []

        if self.use_rbf:
            Ks.append(rbf_kernel(X, Y, self.gamma))
            names.append("RBF")

        if self.use_poly:
            Ks.append(poly_kernel(X, Y, self.degree, self.coef0))
            names.append("Poly")

        if self.use_linear:
            Ks.append(lin_kernel(X, Y))
            names.append("Linear")

        return names, Ks

    # --------------------------------------
    # Fit
    # --------------------------------------
    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64)

        self.scaler_ = None
        X_tr = X

        if self.standardize:
            self.scaler_ = StandardScaler()
            X_tr = self.scaler_.fit_transform(X)

        self.X_train_ = X_tr
        self.y_train_ = np.where(y == y.max(), 1, -1)

        self.kernel_names_, Ks = self._build_kernels(X_tr, X_tr)

        # Learn class-conditional weights
        self.lambda_pos_, self.lambda_neg_ = \
            class_conditional_alignment_weights(Ks, self.y_train_)

        # Build CI-AHK kernel
        K_train = mix_class_aware_kernels(
            Ks, self.lambda_pos_, self.lambda_neg_, self.y_train_
        )

        self.svc_ = SVC(
            C=self.C,
            kernel="precomputed",
            probability=self.probability,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.svc_.fit(K_train, self.y_train_)

        return self

    # --------------------------------------
    # Test kernel
    # --------------------------------------
    def _compute_test_kernel(self, X):
        check_is_fitted(self, ["svc_", "X_train_"])

        Xt = self.scaler_.transform(X) if self.standardize else X
        _, Ks = self._build_kernels(Xt, self.X_train_)

        lambda_mix = 0.5 * (self.lambda_pos_ + self.lambda_neg_)
        K_test = np.zeros_like(Ks[0])

        for m, K in enumerate(Ks):
            K_test += lambda_mix[m] * K

        return K_test

    # --------------------------------------
    # Predict
    # --------------------------------------
    def predict(self, X):
        X = check_array(X, dtype=np.float64)
        K_test = self._compute_test_kernel(X)
        y_pred = self.svc_.predict(K_test)
        return y_pred

    def predict_proba(self, X):
        if not self.probability:
            raise AttributeError("probability=True required")
        X = check_array(X, dtype=np.float64)
        K_test = self._compute_test_kernel(X)
        return self.svc_.predict_proba(K_test)

    def get_kernel_weights(self) -> Dict[str, Dict[str, float]]:
        return {
            "positive_class": dict(zip(self.kernel_names_, self.lambda_pos_)),
            "negative_class": dict(zip(self.kernel_names_, self.lambda_neg_)),
        }
