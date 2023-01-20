import numpy as np


def generate(n=500, nft=5, std=0.0):
    np.random.seed(5)
    n, nft = 500, 5
    x = np.random.rand(n, nft)
    y = x[:, 0] - 5 * x[:, 1] + std * np.random.randn(n)
    return x, y
