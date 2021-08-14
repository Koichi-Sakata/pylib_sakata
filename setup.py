# Copyright (c) 2021 Koichi Sakata

import setuptools

setuptools.setup(
    name="pylib_sakata",
    version="0.0.1",
    author="Example Author",
    author_email="",
    description="Control system design and analysis package",
    url="",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)