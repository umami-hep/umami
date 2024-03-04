## Documentation for Umami

Umami is a framework which can be used for training (most) machine-learning-based taggers used in ATLAS FTAG:

- [DL1](https://gitlab.cern.ch/malanfer/DL1/-/wikis/home)
- [DL1r](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PUBNOTES/ATL-PHYS-PUB-2017-013/)
- [DIPS](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PUBNOTES/ATL-PHYS-PUB-2020-014/)

The intended use is as a software using configuration files to carry out preprocessing, training and evaluation.
It is not written with its use as a library in mind, as its core functionalities are provided by separate python modules which in turn can be installed with `pip`.

These are:

- [ATLAS flavour tagging tools](https://github.com/umami-hep/atlas-ftag-tools)
- [Umami preprocessing](https://github.com/umami-hep/umami-preprocessing)
- [puma – Plotting UMami API](https://github.com/umami-hep/puma)

Umami is also a tagger, the Umami tagger (UT).
Its architecture includes jet features (DL1 inputs) plus a DIPS-like block. The high-level tagger (UT) and the DIPS block are trained in a single training (different to e.g. DL1d developments).

Umami is hosted on CERN GitLab:

- [https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami)

Docker images are available on CERN GitLab container registry and on Docker Hub:

- [CERN GitLab container registry](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/container_registry)
- [Docker Hub](https://hub.docker.com/r/btagging/umami)


An API reference can be found [here](sphinx-docs).

## Tutorials for Umami

A general tutorial is provided as part of the Umami documentation. It makes use of the JetClass dataset and is provided [here](tutorial/index.md).

A corresponding step-by-step tutorial for Umami inside the ATLAS collaboration for jet flavour tagging is provided [here](tutorial/tutorial-umami.md)

Additionally, at the [FTAG Workshop in 2022 in Amsterdam](https://indico.cern.ch/event/1193206/timetable/?view=standard), we gave a tutorial how to work with Umami. You can find the slides together with a recording of the talk [here](https://indico.cern.ch/event/1193206/timetable/?view=standard#b-477082-day-3-afternoon-tutor). Please note that since that tutorial, the Umami software has been developed further and might slightly deviate from the version covered in the tutorial.
