import numpy as np
from scipy.optimize import Bounds
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_array, check_X_y
from ._common import solve_rendered


class KernelBase:
    def _prepare_kernel(self):
        X = self.X_
        if self.kernel == 'rbf':
            self.xsqr_ = xsqr = (X * X).sum(axis=1)
            self.dsqr_ = -2.0 * X.dot(X.T) + \
                xsqr[np.newaxis, :] + xsqr[:, np.newaxis]
        elif self.kernel == 'multi_rbf':
            self.dsqr_ = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2
        elif self.kernel == 'group_multi_rbf':
            dsqr_parts = []
            for g in self.feature_groups:
                xsqr = (X[:, g] * X[:, g]).sum(axis=1)
                dsqr_tmp = -2.0 * \
                    X[:, g].dot(X[:, g].T) + xsqr[np.newaxis, :] + \
                    xsqr[:, np.newaxis]
                dsqr_parts.append(dsqr_tmp)
            self.dsqr_ = np.stack(dsqr_parts, axis=2)
        else:
            raise ValueError(f"unknown kernel '{self.kernel}'")

    def _compute_kernel(self, params, Xother=None):
        if self.kernel == 'rbf':
            if Xother is None:
                if not hasattr(self, 'dsqr_'):
                    self._prepare_kernel()
                dsqr = self.dsqr_
            else:
                xsqr = (Xother * Xother).sum(axis=1)
                dsqr = (
                    -2.0 * self.X_.dot(Xother.T)
                    + self.xsqr_[:, np.newaxis]
                    + xsqr[np.newaxis, :]
                )
            return np.exp(-params['gamma'] * dsqr)
        elif self.kernel == 'multi_rbf':
            if Xother is None:
                if not hasattr(self, 'dsqr_'):
                    self._prepare_kernel()
                dsqr = self.dsqr_
            else:
                dsqr = (self.X_[:, np.newaxis, :] -
                        Xother[np.newaxis, :, :]) ** 2
            return np.exp(-np.tensordot(dsqr, params['gamma'], axes=(2, 0)))
        elif self.kernel == 'group_multi_rbf':
            if Xother is None:
                if not hasattr(self, 'dsqr_'):
                    self._prepare_kernel()
                dsqr = self.dsqr_
            else:
                dsqr_parts = []
                for g in self.feature_groups:
                    xsqr0 = (self.X_[:, g] * self.X_[:, g]).sum(axis=1)
                    xsqr1 = (Xother[:, g] * Xother[:, g]).sum(axis=1)
                    dsqr_tmp = -2.0 * \
                        self.X_[:, g].dot(
                            Xother[:, g].T) + xsqr1[np.newaxis, :] + xsqr0[:, np.newaxis]
                    dsqr_parts.append(dsqr_tmp)
                dsqr = np.stack(dsqr_parts, axis=2)
            return np.exp(-np.tensordot(dsqr, params['gamma'], axes=(2, 0)))
        else:
            raise ValueError(f"unknown kernel '{self.kernel}'")

    def _kernel_deriv(self, idx_0, idx_1, rhs, km):
        if self.kernel == 'rbf':
            return -(km[idx_0, :][:, idx_1] * self.dsqr_[idx_0, :][:, idx_1]).dot(rhs)
        elif self.kernel in {'multi_rbf', 'group_multi_rbf'}:
            return -np.tensordot(
                km[idx_0, :][:, idx_1][:, :, np.newaxis]
                * self.dsqr_[idx_0, :][:, idx_1],
                rhs,
                axes=(1, 0),
            )
        raise ValueError(f"unknown kernel '{self.kernel}'")


class ParameterOptimizationBase:
    def _extract_params(self, p, transform=np.exp):
        if transform is None:
            def transform(x): return x
        if self.kernel == 'rbf':
            return dict(
                lmbda=transform(p[0]),
                gamma=transform(p[1]),
            )
        elif self.kernel in {'multi_rbf', 'group_multi_rbf'}:
            return dict(
                lmbda=transform(p[0]),
                gamma=transform(p[1:]),
            )
        raise ValueError(f"unknown kernel '{self.kernel}'")

    def _compose_params(self, p):
        if self.kernel == 'rbf':
            return np.log([p['lmbda'], p['gamma']])
        elif self.kernel in {'multi_rbf', 'group_multi_rbf'}:
            return np.log(np.concatenate(([p['lmbda']], p['gamma'])))
        raise ValueError(f"unknown kernel '{self.kernel}'")

    def _initialize_parameters(self):
        if self.kernel in {'rbf', 'multi_rbf', 'group_multi_rbf'}:
            gamma0 = self.gamma0
            if self.kernel == 'multi_rbf' and np.isscalar(gamma0):
                nft = self.X_.shape[1]
                gamma0 = gamma0 * np.ones(nft)
            elif self.kernel == 'group_multi_rbf' and np.isscalar(gamma0):
                gamma0 = gamma0 * np.ones(len(self.feature_groups))

            param0 = self._compose_params(
                dict(
                    lmbda=self.lmbda0,
                    gamma=gamma0,
                )
            )
            return param0
        raise ValueError(f"unknown kernel '{self.kernel}'")

    def _get_bounds(self):
        if self.bounds is None:
            return
        if self.kernel in {'rbf', 'multi_rbf', 'group_multi_rbf'}:
            lb_lmbda, ub_lmbda = self.bounds.get('lmbda', (0.0, np.inf))
            lb_gamma, ub_gamma = self.bounds.get('gamma', (0.0, np.inf))
            if self.kernel == 'multi_rbf' and np.isscalar(lb_gamma):
                nft = self.X_.shape[1]
                lb_gamma = lb_gamma * np.ones(nft)
                ub_gamma = ub_gamma * np.ones(nft)
            elif self.kernel == 'group_multi_rbf' and np.isscalar(lb_gamma):
                nft = len(self.feature_groups)
                lb_gamma = lb_gamma * np.ones(nft)
                ub_gamma = ub_gamma * np.ones(nft)
            else:
                lb_gamma = [lb_gamma]
                ub_gamma = [ub_gamma]
            with np.errstate(divide='ignore'):
                return Bounds(
                    np.log(np.concatenate(([lb_lmbda], lb_gamma))),
                    np.log(np.concatenate(([ub_lmbda], ub_gamma))),
                )


class LSSVRBase(KernelBase):
    def _fit_regressor(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        self.X_ = X
        n, nft = X.shape
        self.n_features_in_ = nft
        qm = self._compute_kernel(self.params_)
        qm.flat[:: n + 1] += self.params_['lmbda'] * n

        self.alpha_, self.bias_ = solve_rendered(qm, np.ones(n), y)
        return self

    def predict(self, X):
        if not hasattr(self, 'alpha_'):
            raise NotFittedError()
        X = check_array(X)

        km = self._compute_kernel(self.params_, X)
        return km.T.dot(self.alpha_) + self.bias_
