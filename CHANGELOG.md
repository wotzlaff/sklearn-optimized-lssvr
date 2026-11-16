# Changelog

## 0.0.3

- Final release: the package is deprecated and superseded by
  [optimized-svm](https://wotzlaff.github.io/optimized-svm/).
  `OptimizedLSSVR` corresponds to `optimized_svm.OptimizedSVR(loss="squared")`;
  see the [migration guide](https://wotzlaff.github.io/optimized-svm/migration-lssvr/).
- `import optimized_lssvr` now emits a `DeprecationWarning`.
- Packaging: static version, removed the setuptools_scm leftovers, added
  classifiers (Development Status :: 7 - Inactive).
- Release workflow now publishes only on `v*` tags (or manually).
- No functional changes; see "Known issues" in the README.

## 0.0.2 and earlier

- Initial releases (2021-2023): LSSVR with hyperparameter optimization by
  cross-validation, RBF / multi-RBF / group multi-RBF kernels, sensitivity
  estimation, MIT license, packaging via `pyproject.toml`.
