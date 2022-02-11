"""
This script is used to install the package and all its dependencies. Run

    python setup.py install

to install the package.
"""

from setuptools import setup

setup(
    name="umami",
    version="0.5",  # Also change in module
    packages=[
        "umami",
        "umami.configuration",
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
    include_package_data=True,
    test_suite="umami.tests",
    scripts=[
        "umami/preprocessing.py",
        "umami/train.py",
        "umami/evaluate_model.py",
        "umami/plotting_umami.py",
        "umami/plotting_epoch_performance.py",
    ],
    data_files=["umami/configs/global_config.yaml"],
    description="Machine learning based flavour tagging training framework.",
    url="https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami",
)
