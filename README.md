[![pipeline status](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/badges/master/pipeline.svg)](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/commits/master) [![coverage report](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/badges/master/coverage.svg)](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/commits/master) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Umami

The Umami documentation is avaliable here:

[![Umami docs](https://img.shields.io/badge/info-documentation-informational)](https://umami-docs.web.cern.ch/umami-docs/)

Below is included a brief summary on how to get started fast.

## Installation

### Docker image

```bash
singularity exec docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami:latest bash
```

besides the CPU image, there is also a GPU image available which is especially useful for the training step

```bash
singularity exec --nv docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami:latest-gpu bash
```

This image has Tensorflow installed for training the taggers. Another option is PyTorch. You can use it with this:

```bash
singularity exec --nv docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umamibase:latest-pytorch-gpu bash
```

If you want to change something in the code (outside of config files), you need to run

```bash
source run_setup.sh
```

which sources the [run_setup.sh](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/run_setup.sh). Otherwise, the already in the image installed version of Umami is used.

### Manual setup

Alternatively you can also check out this repository via `git clone` and then run

```bash
python setup.py install
```

this will install the umami package

If you want to modify the code you should run instead

```bash
python setup.py develop
```

which creates a symlink to the repository.

If you want to commit changes it is recommended to install the pre-commit hooks by doing the following:

```bash
pre-commit install
```

This will run isort, black and flake8 on staged python files when commiting

## Testing & Linter

The test suite can be run via

```bash
pytest ./umami/tests/ -v
```

If you want to only run, for example, the unit tests for the evaluation tools, this can be done via

```bash
pytest ./umami/tests/unit/evaluation_tools/ -v
```

To run the integration tests, you need to run them in the correct order: preprocessing, training, plotting.   
Otherwise, you will get an error that some files are missing. You can run those via

```bash
pytest ./umami/tests/integration/test_preprocessing.py -v
```

In order to run the code style checker `flake8` use the following command

```bash
flake8 ./umami
```

## Preprocessing

For the training of umami the ntuples are used as specified in the section [MC Samples](#mc-samples).

The ntuples need to be preprocessed following the [preprocessing instructions](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/preprocessing.md).

## DL1r instructions

If you want to train or evaluate DL1r please follow the [DL1r-instructions](docs/DL1r-instructions.md).

## DIPS instructions

If you want to train or evaluate DIPS please follow the [DIPS-instructions](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/Dips-instructions.md)
