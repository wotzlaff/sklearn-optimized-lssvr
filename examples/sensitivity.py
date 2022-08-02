import numpy as np
from optimized_lssvr import OptimizedLSSVR, estimate_sensitivity


def main():
    np.random.seed(5)
    n, nft = 500, 5
    x = np.random.rand(n, nft)
    sigma = 0.0
    y = x[:, 0] - 5 * x[:, 1] + sigma * np.random.randn(n)

    model = OptimizedLSSVR(lmbda0=1.0, gamma0=1.0, kernel='rbf', verbose=1)
    model.fit(x, y)
    s = estimate_sensitivity(model, x)
    coeffs = s.mean(axis=0)
    print(coeffs)


if __name__ == '__main__':
    main()
