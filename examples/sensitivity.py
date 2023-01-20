import data
from optimized_lssvr import OptimizedLSSVR, estimate_sensitivity


def main():
    x, y = data.generate()
    model = OptimizedLSSVR(lmbda0=1.0, gamma0=1.0, kernel='rbf', verbose=1)
    model.fit(x, y)
    s = estimate_sensitivity(model, x)
    coeffs = s.mean(axis=0)
    print(coeffs)


if __name__ == '__main__':
    main()
