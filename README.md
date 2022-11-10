[![pipeline status](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/badges/master/pipeline.svg)](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/commits/master) [![coverage report](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/badges/master/coverage.svg)](https://umami-ci-coverage.web.cern.ch/master/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Umami

The Umami documentation is avaliable here:

[![Umami docs](https://img.shields.io/badge/info-documentation-informational)](https://umami.docs.cern.ch/)

Below is included a brief summary on how to get started fast.

## Installation
You can find the detailed described in the [umami docs](https://umami.docs.cern.ch/setup/installation/).


## Testing & Linter

To better collaborate on this project, we require some code practices such as:
- linting (`flake8`, `yamllint`)
- unit tests
- integration tests

More details can be found [here](https://umami.docs.cern.ch/setup/development/).


## Preprocessing

For the training of umami the ntuples are used as specified in the section [MC Samples](https://ftag.docs.cern.ch/software/samples/).

The ntuples need to be preprocessed following the [Preprocessing Instructions](https://umami-docs.web.cern.ch/preprocessing/Overview/).

## Training

If you want to train or evaluate DL1r or DIPS please follow the [Training Instructions](https://umami-docs.web.cern.ch/trainings/Overview/).

## Tutorial for Umami

At the [FTAG Workshop in 2022 in Amsterdam](https://indico.cern.ch/event/1193206/timetable/?view=standard), we gave a tutorial how to work with Umami. You can find the slides together with a recording of the talk [here](https://indico.cern.ch/event/1193206/timetable/?view=standard#b-477082-day-3-afternoon-tutor). The corresponding step-by-step tutorial can be found in the FTAG docs webpage [here](https://ftag.docs.cern.ch/software/tutorials/tutorial-umami/).
