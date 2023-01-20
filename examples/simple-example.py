import sklearn.datasets
from optimized_lssvr import OptimizedLSSVR


def main():
    data = sklearn.datasets.load_diabetes()
    x = data['data']
    y = data['target']

    model0 = OptimizedLSSVR(
        verbose=1,
    )
    model0.fit(x, y)
    p = model0.params_
    print('final relative MSE:', model0.relative_mse_)
    print('optimized parameters:', p)

    model1 = OptimizedLSSVR(
        kernel='multi_rbf',
        lmbda0=p['lmbda'],
        gamma0=p['gamma'],
        verbose=1,
    )
    model1.fit(x, y)
    print('final relative MSE:', model1.relative_mse_)
    print('optimized parameters:', model1.params_)


if __name__ == '__main__':
    main()
