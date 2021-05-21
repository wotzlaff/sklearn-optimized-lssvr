import numpy as np
import scipy.optimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_X_y
from ._bases import LSSVRBase, ParameterOptimizationBase
from ._common import solve_rendered


class OptimizedLSSVR(BaseEstimator, RegressorMixin, LSSVRBase, ParameterOptimizationBase):
    def __init__(
        self, kernel='rbf', gamma0=1.0, lmbda0=1e-3, n_splits=3, method=None, tol=1e-6
    ):
        self.kernel = kernel
        self.gamma0 = gamma0
        self.lmbda0 = lmbda0
        self.n_splits = n_splits
        self.tol = tol
        self.method = method

    def optimize_parameters(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        self.X_ = X
        self.ysqr_ = (y ** 2).sum()

        n = X.shape[0]
        self.kf_ = KFold(self.n_splits)
        self._prepare_kernel()

        param0 = self._initialize_parameters()
        n_param = param0.size

        def objective(p):
            params = self._extract_params(p)
            km = self._compute_kernel(params)
            err = np.zeros(n)
            dmse = np.zeros(n_param)
            for idx_tr, idx_val in self.kf_.split(X):

                k_tr = km[idx_tr, :][:, idx_tr]
                qm = k_tr.copy()
                n_tr = qm.shape[0]
                qm.flat[:: n_tr + 1] += params['lmbda']

                a, b = solve_rendered(
                    qm,
                    np.ones(n_tr),
                    y[idx_tr],
                )

                k_val = km[idx_tr, :][:, idx_val]
                f_val = k_val.T.dot(a) + b
                n_val = f_val.size
                err_val = f_val - y[idx_val]
                err[idx_val] = err_val

                rhs = k_val.dot(2.0 * err_val)
                rhs0 = 2.0 * err_val.sum()
                da, db = solve_rendered(qm, np.ones(n_tr), rhs, rhs0)
                # lmbda deriv
                dmse[0] -= da.dot(a)

                # gamma deriv
                q = self._kernel_deriv(
                    idx_tr, idx_val, 2.0 * err_val, km
                ) - self._kernel_deriv(idx_tr, idx_tr, da, km)
                dmse[1:] += np.tensordot(q, a, axes=(0, 0))

            mse = (err * err).sum() / self.ysqr_
            dmse = dmse * np.exp(p) / self.ysqr_
            return mse, dmse

        # solve hpo problem
        res = scipy.optimize.minimize(
            objective,
            param0,
            jac=True,
            tol=self.tol,
            method=self.method,
        )
        param1 = res.x
        self.params_ = self._extract_params(param1)
        self.dparams_ = self._extract_params(res.jac, transform=None)
        return self.params_

    def fit(self, X, y, optimize_parameters=True):
        if optimize_parameters:
            self.optimize_parameters(X, y)
        self._fit_regressor(X, y)
        return self
