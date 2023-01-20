# optimized-lssvr - A basic implementation of LS-SVR with optimized hyperparameters
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

## References
<a id="1">[1]</a>
Fischer, A., Langensiepen, G., Luig, K., Strasdat, N., & Thies, T. (2015). Efficient optimization of hyper-parameters for least squares support vector regression. Optimization Methods and Software, 30(6), 1095-1108.