from ._bases import LSSVRBase
from sklearn.base import BaseEstimator, RegressorMixin


class LSSVR(BaseEstimator, RegressorMixin, LSSVRBase):
    """ ...

    ...

    Parameters
    ----------
    kernel : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    gamma : float | np.array
        Kernel scaling parameter
    lmbda : float
        Regularization parameter

    Examples
    --------
    >>> from optimized_lssvr import LSSVR
    >>> import numpy as np
    >>> X = np.arange(10).reshape(10, 1)
    >>> y = np.zeros((10, ))
    >>> estimator = LSSVR()
    >>> estimator.fit(X, y)
    LSSVR()
    """

    def __init__(self, kernel='rbf', gamma=1.0, lmbda=1e-3, feature_groups=None):
        self.kernel = kernel
        self.gamma = gamma
        self.lmbda = lmbda
        self.feature_groups = feature_groups

    def fit(self, X, y):
        self.params_ = dict(gamma=self.gamma, lmbda=self.lmbda)
        return self._fit_regressor(X, y)
