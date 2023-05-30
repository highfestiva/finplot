# -*- coding: utf-8 -*-
import setuptools
from finplot._version import __version__

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='finplot',
    version=__version__,
    author='Jonas Byström',
    author_email='highfestiva@gmail.com',
    description='Finance plotting',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/highfestiva/finplot',
    packages=['finplot'],
    install_requires=['numpy>=1.23.5', 'pandas>=1.5.2', 'PyQt6>=6.4.0', 'pyqtgraph>=0.13.1', 'python-dateutil'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
