"""
This script is used to install the package and all its dependencies. Run

    python setup.py install

to install the package.
"""

from setuptools import setup


setup(name='umami',
      version='0.0.0',  # Also change in module
      packages=["umami", "umami.tests"],
      install_requires=["h5py",
                        "numpy",
                        "matplotlib",
                        "seaborn",
                        "tables",
                        "pandas",
                        "tensorflow",
                        "keras"],
      test_suite='umami.tests',
      description='Machine learning based flavour tagging training framework.',
      url="https://gitlab.cern.ch/mguth/umami"
      )
