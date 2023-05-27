from setuptools import setup, find_packages

setup(
    name='teras',
    version='0.1.0',
    description='A Unified Deep Learning Framework for Tabular Data.',
    author='Khawaja Abaid',
    author_email='khawaja.abaid@gmail.com',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'tensorflow-addons',
        'tensorflow-probability',
        'pandas',
        'scikit-learn',
        'numpy',
    ],
)