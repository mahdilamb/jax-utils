import pathlib
from typing import List
from setuptools import setup, find_packages

setup(
    name='jax_utils',
    version='0.0.1',
    packages=find_packages(include=['jax_utils', 'jax_utils.*']),
    install_requires=[
        'jax',
    ],
)