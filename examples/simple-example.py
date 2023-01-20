import sklearn.datasets
from optimized_lssvr import OptimizedLSSVR


def main():
    data = sklearn.datasets.load_diabetes()
    x = data['data']
    y = data['target']

    model = OptimizedLSSVR(verbose=1)
    model.fit(x, y)

    print('final relative MSE:', model.relative_mse_)
    print('optimized parameters:', model.params_)


if __name__ == '__main__':
    main()
