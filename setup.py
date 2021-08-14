# Copyright (c) 2021 Koichi Sakata

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pylib_sakata",
    version="0.0.1",
    author="Koichi Sakata",
    author_email="",
    description="Control system design and analysis package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Koichi-Sakata/pylib_sakata",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)