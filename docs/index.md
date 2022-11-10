## Documentation for Umami

Umami is a framework which can be used for training (most) machine-learning-based taggers used in ATLAS FTAG:

- [DL1](https://gitlab.cern.ch/malanfer/DL1/-/wikis/home)
- [DL1r](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PUBNOTES/ATL-PHYS-PUB-2017-013/)
- [DIPS](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PUBNOTES/ATL-PHYS-PUB-2020-014/)

Umami is also a tagger, the Umami tagger (UT).
Its architecture includes jet features (DL1 inputs) plus a DIPS-like block. The high-level tagger (UT) and the DIPS block are trained in a single training (different to e.g. DL1d developments).

Umami is hosted on CERN GitLab:

- [https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami)

Docker images are available on CERN GitLab container registry and on Docker Hub:

- [CERN GitLab container registry](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/container_registry)
- [Docker Hub](https://hub.docker.com/r/btagging/umami)


An API reference can be found [here](sphinx-docs).

## Tutorial for Umami

At the [FTAG Workshop in 2022 in Amsterdam](https://indico.cern.ch/event/1193206/timetable/?view=standard), we gave a tutorial how to work with Umami. You can find the slides together with a recording of the talk [here](https://indico.cern.ch/event/1193206/timetable/?view=standard#b-477082-day-3-afternoon-tutor). The corresponding step-by-step tutorial can be found in the FTAG docs webpage [here](https://ftag.docs.cern.ch/software/tutorials/tutorial-umami/)
