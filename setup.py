#! /usr/bin/env python
"""A basic implementation of LS-SVR with optimized hyperparameters."""

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('optimized_lssvr', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'optimized-lssvr'
DESCRIPTION = 'Least Squares Support Vector Regression with optimized hyperparameters'
with codecs.open('README.md', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Nico Strasdat'
MAINTAINER_EMAIL = 'nstrasdat@gmail.com'
URL = 'https://github.com/wotzlaff/sklearn-optimized-lssvr'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/wotzlaff/sklearn-optimized-lssvr'
VERSION = __version__
INSTALL_REQUIRES = ['numpy', 'scipy', 'scikit-learn']
CLASSIFIERS = []
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
