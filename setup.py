"""
@author: gjorando
"""

import os
import importlib
import sys
from setuptools import setup, find_packages


def read(*tree):
    """
    Read a file from the setup.py location.
    """

    full_path = os.path.join(os.path.dirname(__file__), *tree)
    with open(full_path, encoding='utf-8') as file:
        return file.read()


def version(main_package):
    """
    Read the version number from the __version__ variable in the main
    package __init__ file.
    """

    package = "{}.__init__".format(main_package)
    init_module = importlib.import_module(package)
    try:
        return init_module.__version__
    except AttributeError:
        raise RuntimeError("No version string found in {}.".format(package))


def requirements(*tree):
    """
    Read the requirements list from a requirements.txt file.
    """

    requirements_file = read(*tree)
    return [r for r in requirements_file.split("\n") if r != ""]


def long_description(*tree):
    """
    setup.py only supports .rst files for the package description. As a
    result, we need to convert README.md on the fly.
    """

    try:
        from pypandoc import convert_file
        tree_join = os.path.join(os.path.dirname(__file__), *tree)
        rst_readme = convert_file(tree_join, 'rst')
        rst_path = "{}.rst".format(os.path.splitext(tree_join)[0])
        with open(rst_path, "w") as file:
            file.write(rst_readme)
        return rst_readme
    except ImportError:
        sys.stderr.write(
            "warning: pypandoc module not found,"
            "README.md couldn't be converted.\n"
        )
        return read(*tree)


setup(
    name="neurartist",
    version=version("neurartist"),
    author="Guillaume Jorandon",
    description="Ready-to-use artistic deep learning algorithms",
    long_description=long_description("README.md"),
    url="https://github.com/gjorando/style-transfer",
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements("requirements.txt"),
    entry_points={
        'console_scripts': ['neurartist=neurartist.cli:main']
    },
    python_requires='>=3',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Artistic Software',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)
