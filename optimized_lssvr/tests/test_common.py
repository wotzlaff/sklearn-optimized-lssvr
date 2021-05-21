import pytest

from sklearn.utils.estimator_checks import check_estimator

from optimized_lssvr import LSSVR
from optimized_lssvr import OptimizedLSSVR


@pytest.mark.parametrize(
    "Estimator", [LSSVR, OptimizedLSSVR]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator())
