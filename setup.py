#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 2023
@author: Simon Pelletier
"""


from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='progressa',
    version='0.1',
    packages=['progressa', 'progressa.analysis', 'progressa.create_images', 'progressa.train_models'],
    url='',
    license='',
    author='',
    author_email='',
    description='',
    install_requires=requirements
)
