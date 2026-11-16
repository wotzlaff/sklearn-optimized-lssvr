# Least Squares Support Vector Regression with optimized hyperparameters

> **Deprecated.** This package is no longer maintained and has been superseded by
> [`optimized-svm`](https://github.com/wotzlaff/optimized-svm)
> ([documentation](https://wotzlaff.github.io/optimized-svm/)).
> See the [migration guide](https://wotzlaff.github.io/optimized-svm/migration-lssvr/).
>
> ```python
> # before
> from optimized_lssvr import OptimizedLSSVR
> model = OptimizedLSSVR()
> # after
> from optimized_svm import OptimizedSVR
> model = OptimizedSVR(loss="squared")
> ```

This is a simple implementation of the hyperparameter optimization approach proposed in [[1]](#1).

## Installation
```sh
pip install optimized-lssvr
```

## Example
```python
import numpy as np
from optimized_lssvr import OptimizedLSSVR

# generate example data
n, nft = 500, 5
x = np.random.rand(n, nft)
y = x[:, 0] - 5 * x[:, 1]

# create and fit model
model = OptimizedLSSVR(verbose=1)
model.fit(x, y)
print('final relative MSE:', model.relative_mse_)
print('optimized parameters:', model.params_)
```
More examples can be found in the [`examples`](examples) directory.

## Known issues
This final release (0.0.3) only adds the deprecation notice; the following issues will not be fixed.
- The final fit adds `lmbda * n` to the kernel diagonal, whereas the cross-validation uses `lmbda * n_tr`
  (`n_tr` = size of the training fold). The refitted model is therefore regularized more strongly than the
  model the hyperparameters were optimized for (by about 25 % with 5 folds).
- The cross-validation folds are not shuffled.
- `sklearn.utils.estimator_checks.check_estimator` fails with current scikit-learn versions.

## References
<a id="1">[1]</a>
Fischer, A., Langensiepen, G., Luig, K., Strasdat, N., & Thies, T. (2015). Efficient optimization of hyper-parameters for least squares support vector regression. Optimization Methods and Software, 30(6), 1095-1108.