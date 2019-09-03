from setuptools import setup, find_packages

setup(
    name = "Veril",
    version = "0.0.1",
    author = "Shen Shen",
    url = "http://github.com/shensquared/Veril",
    author_email = "shenshen@mit.edu",
    packages = find_packages(),
    install_requires=[
        "tensorflow<=1.14",
        "keras==2.2.4"
    ],
)
