"""
This script is used to install the package and all its dependencies. Run

    python -m pip install .

to install the package.
"""

from setuptools import setup

setup(
    name="umami",
    version="0.6",  # Also change in module
    packages=[
        "umami",
        "umami.classification_tools",
        "umami.configuration",
        "umami.data_tools",
        "umami.evaluation_tools",
        "umami.helper_tools",
        "umami.input_vars_tools",
        "umami.metrics",
        "umami.models",
        "umami.plotting",
        "umami.preprocessing_tools",
        "umami.tf_tools",
        "umami.tools.PyATLASstyle",
        "umami.tools",
        "umami.train_tools",
    ],
    test_suite="umami.tests",
    include_package_data=True,
    scripts=[
        "umami/preprocessing.py",
        "umami/train.py",
        "umami/evaluate_model.py",
        "umami/plotting_umami.py",
        "umami/plotting_epoch_performance.py",
    ],
    description="Machine learning based flavour tagging training framework.",
    url="https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami",
)
