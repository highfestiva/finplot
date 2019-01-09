import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='finplot',
    version='0.1.3',
    author='Jonas Byström',
    author_email='highfestiva@gmail.com',
    description='Finance plotting',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/highfestiva/finplot',
    packages=['finplot'],
    install_requires=['pandas>=0.23.4', 'PyQt5==5.9', 'pyqtgraph>=0.10.0'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
