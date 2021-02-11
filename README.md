# Umami

The Umami documentation is avaliable here:

[![Umami docs](https://img.shields.io/badge/info-documentation-informational)](https://umami-docs.web.cern.ch/umami-docs/)
[![coverage report](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/badges/master/coverage.svg)](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/commits/master)


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

## Testing & Linter

The test suite can be run via

```bash
pytest ./umami/tests/ -v
```

If you want to only run unit tests, this can be done via

```bash
pytest ./umami/tests/unit/ -v
```

and the integration test similarly via

```bash
pytest ./umami/tests/integration/ -v
```

In order to run the code style checker `flake8` use the following command

```bash
flake8 ./umami
```

## DL1r instructions

If you want to train or evaluate DL1r please follow the [DL1r-instructions](docs/DL1r-instructions.md).

## DIPS instructions

If you want to train or evaluate DIPS please follow the [DIPS-instructions](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/Dips-instructions.md)

## Preprocessing

For the training of umami the ntuples are used as specified in the section [MC Samples](#mc-samples).

The ntuples need to be preprocessed following the [preprocessing instructions](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/preprocessing.md).
