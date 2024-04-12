from setuptools import setup, find_packages

setup(
    name='teras',
    version='0.3.1',
    description='A Unified Deep Learning Library for Tabular Data.',
    author='Khawaja Abaid',
    author_email='khawaja.abaid@gmail.com',
    packages=find_packages(),
    install_requires=[
        'keras',
        'pandas',
        'scikit-learn',
        'numpy',
    ],
)