from setuptools import setup, find_packages

setup(
    name = 'src',
    packages = find_packages(),
    version = '0.1.0',
    requires=['einops'],
)