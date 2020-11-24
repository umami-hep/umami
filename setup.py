"""
This script is used to install the package and all its dependencies. Run

    python setup.py install

to install the package.
"""

from setuptools import setup


setup(name='umami',
      version='0.0.0',  # Also change in module
      packages=["umami", "umami.tests", "umami.train_tools", "umami.tools",
                "umami.preprocessing_tools", "umami.tools.PyATLASstyle",
		"umami.evaluation_tools"],
      #install_requires=["h5py",
       #                 "numpy",
        #                "matplotlib",
         #               "seaborn",
          #              "tables",
           #             "pandas",
                        #"tensorflow",
                        #"keras"
            #            ],
      include_package_data=True,
      test_suite='umami.tests',
      scripts=["umami/preprocessing.py",
               "umami/train_DL1.py", "umami/train_Dips.py",
               "umami/train_umami.py"],
      description='Machine learning based flavour tagging training framework.',
      url="https://gitlab.cern.ch/mguth/umami"
      )
