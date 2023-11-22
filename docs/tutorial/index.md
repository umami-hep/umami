This tutorial introduces how to use Umami for more general applications than just jet flavour tagging in the ATLAS experiment.

The basis of this tutorial is the [JetClass](https://zenodo.org/records/6619768) dataset.
It is described in the paper [Particle Transformer for Jet Tagging](https://arxiv.org/abs/2202.03772) and available here: [https://doi.org/10.5281/zenodo.6619768](https://zenodo.org/records/6619768).

Before the dataset can be used for Umami, it needs to be converted in the `h5` format with the data structure which is assumed for Umami. 

- The preprocessing tutorial section explains how to retrieve the dataset, convert it, and run the umami preprocessing.
- The training tutorial section explains how to run and monitor the training of networks for classification.
- The validation tutorial section explains how to validate the trained model and investigate loss and accuracy as function of training epoch.
