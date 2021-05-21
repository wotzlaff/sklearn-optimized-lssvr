import pytest

from sklearn.datasets import load_iris

from optimized_lssvr import LSSVR
from optimized_lssvr import OptimizedLSSVR


@pytest.fixture
def data():
    X, y = load_iris(return_X_y=True)
    return X[:50], y[:50]


def test_simple_lssvr(data):
    X, y = data
    clf = LSSVR()
    assert clf.gamma == 1.0
    assert clf.lmbda == 1.0e-3

    clf.fit(X, y)
    assert hasattr(clf, 'alpha_')
    assert hasattr(clf, 'bias_')
    assert hasattr(clf, 'X_')

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
