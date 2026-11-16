import warnings

from ._simple_lssvr import LSSVR
from ._optimized_lssvr import OptimizedLSSVR
from ._sensitivity import estimate_sensitivity

from ._version import __version__

__all__ = ['LSSVR', 'OptimizedLSSVR', 'estimate_sensitivity', '__version__']

warnings.warn(
    "optimized-lssvr is deprecated and no longer maintained. It is superseded "
    "by optimized-svm (https://wotzlaff.github.io/optimized-svm/); "
    "OptimizedLSSVR corresponds to optimized_svm.OptimizedSVR(loss='squared'). "
    "See https://wotzlaff.github.io/optimized-svm/migration-lssvr/ "
    "for migration instructions.",
    DeprecationWarning,
    stacklevel=2,
)
