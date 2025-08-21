#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 2021
@author: Simon Pelletier
"""


from setuptools import setup, find_packages


setup(
    name="msml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "tensorflow",
        "torch",
        "xgboost",
        "msalign",
        "ax-platform",
    ],
    python_requires=">=3.8",
)