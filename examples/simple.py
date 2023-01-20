import numpy as np
from optimized_lssvr import OptimizedLSSVR


def main():
    # generate example data
    n, nft = 500, 5
    x = np.random.rand(n, nft)
    y = x[:, 0] - 5 * x[:, 1]

    # create and fit model
    model = OptimizedLSSVR(verbose=1)
    model.fit(x, y)
    print('final relative MSE:', model.relative_mse_)
    print('optimized parameters:', model.params_)


if __name__ == '__main__':
    main()
