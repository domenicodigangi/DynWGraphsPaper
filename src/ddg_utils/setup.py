

#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

requirements = ['Click>=7.0', 'numpy>=1.19', 'pandas>=1.2' ]

test_requirements = ['pytest>=3', ]

setup(
    author="Domenico Di Gangi",
    author_email='digangidomenico@gmail.com',
    python_requires='>=3.6',
    install_requires=requirements,
    license="MIT license",
    include_package_data=True,
    keywords='ddg_utils',
    name='ddg_utils',
    packages=find_packages(include=['ddg_utils', 'ddg_utils.*']),
    version='0.1.0',
    zip_safe=False,
)
