import numpy as np


def generate(n=500, nft=5):
    np.random.seed(5)
    n, nft = 500, 5
    x = np.random.rand(n, nft)
    sigma = 0.0
    y = x[:, 0] - 5 * x[:, 1] + sigma * np.random.randn(n)
    return x, y
