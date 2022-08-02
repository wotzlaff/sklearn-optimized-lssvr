import numpy as np


def estimate_sensitivity(self, X):
    km = self._compute_kernel(self.params_, X)
    d = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    gamma = self.params_['gamma']
    gamma = gamma if isinstance(gamma, float) else np.array(gamma)[np.newaxis, :]
    dka = -2.0 * gamma * np.tensordot(
        km[:, :, np.newaxis] * d, self.alpha_,
        axes=(1, 0),
    )
    return dka
