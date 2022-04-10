# -*- coding: utf-8 -*-
import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='finplot',
    version='1.8.1',
    author='Jonas Byström',
    author_email='highfestiva@gmail.com',
    description='Finance plotting',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/highfestiva/finplot',
    packages=['finplot'],
    install_requires=['pandas', 'PyQt5', 'pyqtgraph>=0.11.1', 'python-dateutil'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
