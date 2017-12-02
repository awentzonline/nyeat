#!/usr/bin/env python
from distutils.core import setup
from setuptools import find_packages


setup(
    name='nyeat',
    version='0.0.1',
    description='NEAT implemented with numpy and networkx.',
    author='Adam Wentz',
    author_email='adam@adamwentz.com',
    url='https://github.com/awentzonline/nyeat/',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'six',
        'networkx'
    ]
)
