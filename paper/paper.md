---
title: 'Umami: A Python toolkit for jet flavour tagging in the ATLAS experiment'
tags:
  - Python
  - Dockerfile
  - high energy physics
  - jet physics
  - flavour tagging
  - machine learning
authors:
  - name: Jackson Barr
    orcid: 0000-0002-9752-9204 
    equal-contrib: true
    affiliation: 5
  - name: Joschka Birk
    orcid: 0000-0002-1931-0127
    equal-contrib: true
    affiliation: 1
  - name: Maxence Draguet
    equal-contrib: true
    affiliation: 4
  - name: Stefano Franchellucci
    orcid: 0000-0003-0695-0798
    equal-contrib: true
    affiliation: 2
  - name: Alexander Froch
    orcid: 0000-0002-8259-2622
    equal-contrib: true
    affiliation: 1
  - name: Philipp Gadow
    orcid: 0000-0003-4475-6734
    equal-contrib: true
    affiliation: 3
  - name: Manuel Guth
    orcid: 0000-0002-6647-1433
    equal-contrib: true
    affiliation: 2
  - name: Osama Karkout
    orcid: 0000-0002-4907-9499
    equal-contrib: true
    affiliation: 9
  - name: Dmitrii Kobylianskii
    orcid: 0009-0002-0070-5900
    equal-contrib: true
    affiliation: 10
  - name: Ivan Oleksiyuk
    orcid: 0000-0002-4784-6340
    equal-contrib: true
    affiliation: 2
  - name: Nikita Ivvan Pond
    orcid: 0000-0002-5966-0332
    equal-contrib: true
    affiliation: 5
  - name: Frederic Renner
    orcid: 0000-0002-9475-3075
    equal-contrib: true
    affiliation: 3
  - name: Sebastien Rettie
    orcid: 0000-0002-7092-3893
    equal-contrib: true
    affiliation: 7
  - name: Victor Hugo Ruelas Rivera
    orcid: 0000-0002-2116-048X
    equal-contrib: true
    affiliation: 6
  - name: Tomke Schröer
    orcid: 0000-0001-7967-6385
    equal-contrib: true
    affiliation: 2
  - name: Samuel Van Stroud
    orcid: 0000-0002-7969-0301
    equal-contrib: true
    affiliation: 5
  - name: Janik Von Ahnen
    orcid: 0000-0003-4032-0079
    equal-contrib: true
    affiliation: 3
  - name: Martino Tanasini
    orcid: 0000-0002-6313-4175
    equal-contrib: true
    affiliation: 8
  
  
  
affiliations:
 - name: Albert-Ludwigs-Universität Freiburg, Germany
   index: 1
 - name: Université de Genève, Switzerland
   index: 2
 - name: Deutsches Elektronen-Synchrotron DESY, Germany
   index: 3
 - name: University of Oxford, United Kingdom
   index: 4
 - name: University College London, United Kingdom
   index: 5
 - name: Humboldt University Berlin, Germany
   index: 6
 - name: European Laboratory for Particle Physics CERN, Switzerland
   index: 7
 - name: INFN Genova and Universita' di Genova, Italy
   index: 8
 - name: Nikhef National Institute for Subatomic Physics and University of Amsterdam, Netherlands
   index: 9
 - name: Department of Particle Physics and Astrophysics, Weizmann Institute of Science, Israel
   index: 10
date: 01 May 2023
bibliography: paper.bib

---

# Summary

Flavour-tagging, the identification of jets originating from bottom and charm quarks, is a critically important technique in the data analysis of the ATLAS experiment [@ATLAS:2008] at the Large Hadron Collider [@Evans:2008]. It is applied in precision measurements of the Standard Model and the Higgs boson, as well as in searches for yet unknown phenomena.
The long lifetime, high mass, and large decay multiplicity of hadrons containing bottom and charm quarks provide distinct signatures of charged particle trajectories in the detector which can be exploited by machine learning algorithms.
The excellent knowledge of the detector and the physics processes at hand enables simulations to provide high-quality training data for algorithms.
The `Umami` software toolkit provides a unified data pipeline, definition of the algorithms, training and performance evaluation with a high degree of automation.

# Statement of need

`Umami` is a Python [@Rossum:2009] toolkit for training and evaluating machine learning algorithms used in high-energy-physics jet flavour tagging.
The creation and training of production-grade machine learning models is supported by the `TensorFlow` [@tensorflow:2015] and `keras` [@chollet:2015] packages for Python. The training datasets feature highly imbalanced distributions among the target classes and input features of vastly different magnitude. Consequentially, the preprocessing of the training data requires resampling to provide balanced datasets and normalisation of the input features by scaling and shifting.

`Umami` provides a class-based and user-friendly interface with `yaml` configuration files to steer the data preprocessing and the training of deep neural networks. It is deployed as a Python module which can be installed with `setuptools` or used via Docker images [@Merkel:2014]. `Umami` was designed to be used by researchers in the ATLAS collaboration and is open to be applied in more general context.

# Related work

The application of machine learning in high energy physics, particularly for the classification of jets, is a common and critically important technique [@Guest:2018; @Cagnotta:2022]. In contrast to previous efforts in jet flavour tagging [@Bols:2020; @ATLAS:2019], the current state-of-the-art algorithms [@Qu:2022] rely on specialised toolkits, such as the `Weaver` framework [@Qu:2020]. These toolkits enable the design of algorithms by taking care of input processing, steering the training on large datasets and providing performance metrics with fast inference.

# Development Notes

The development of the package is based on PEP8 standards. They are enforced by a continuous integration pipeline in a GitLab project, using the `flake8` linter and the `black` command-line tool for code formatting. The code quality is tested as part of the continuous integration pipeline with the `pytest` module, using unit tests and integration tests.
Documentation of the software is built automatically with the `mkdocs` and `sphinx` modules and deployed to a website.
The `Umami` toolkit has been released as open-source software under the Apache v2 license. 

# Software description

The `Umami` toolkit provides an integrated workflow including input data preprocessing, algorithm training, and performance evaluation. Furthermore, it interfaces to `lwtnn` [@Guest:2022] to export the trained models to `json` files for `C++` deployment in the ATLAS software stack [@ATLAS:2021].

## Preprocessing

The algorithms are trained on simulated physics processes which provide jets originating from bottom and charm quarks, as well as the background processes which produce jets originating from other sources. Several datasets with different physics processes can be combined to a hybrid sample, which is populated over a large momentum range.
The classes in the input dataset are highly imbalanced. Consequentially, `Umami` provides under- and oversampling methods as well as a weighting method to ensure similar kinematic distributions for the jets of different target classes.
The range of values of the input features on which the algorithm is trained can differ considerably. Consequentially, `Umami` normalises the range of the variables used in training and creates a `json` file with scaling and shifting parameters.
The resulting training data has balanced target classes and normalised input features. It can be stored in either as an `hdf5` file [@hdf5:2023] or in the binary `TFRecords` format to improve reading speed provided by `TensorFlow`.
The steps involved in the preprocessing workflow are illustrated in \autoref{fig:preprocessing}.
First, datasets which are pure in the target classes are extracted from the simulated physics processes in the "Preparation" step. Then, the training datasets are resampled in the "Resampling" step and the input features are scaled and shifted to ensure normalised distributions in the "Scaling/Shifting" step. Finally, the training sample is written to disk, together with the "Scale Dict" and datasets for validation and performance evaluation.

![Illustration of the preprocessing workflow in `Umami`.\label{fig:preprocessing}](preprocessing.png){ width=60% }

## Training

Different architectures of neural networks, including Deep Multi-Layer-Perceptrons [@LeCun:2015] and Deep Sets [@Zaheer:2017], are supported in `Umami` for definition with configuration files.
The training is performed with `TensorFlow` using the `keras` back-end and the Adam optimizer [@Kingma:2015], supporting the use of GPU resources to drastically shorten the time to train the networks.
Parameters defined in the configuration file include the batch size, the number of epochs, as well as the learning rate.
The resulting model from each epoch during the training is saved. These models are evaluated on a validation dataset to identify the optimal configuration. Typical performance metrics include the validation loss and the efficiency in identifying the correct jet labels. These can be plotted as a function of the training epoch to select the epoch which corresponds to the optimal performance of the trained model.
The steps involved in the training workflow are illustrated in \autoref{fig:training}. After the "Training" step, the optimal model configuration is chosen in the "Validation" step by evaluating the trained model with the "Scale Dict" on a validation sample.

![Illustration of the training workflow in `Umami`.\label{fig:training}](training.png){ width=60% }


## Performance evaluation


The performance of the chosen model can be evaluated in publication-grade plots, which are steered with configuration files. The plots are created using the `matplotlib` [@Hunter:2007] and `puma` [@Birk:2023] Python modules.
Typical performance plots include

- receiver-operator-characteristics (ROC),
- efficiency of the signal class and rejection of background classes as functions of certain variables,
- confusion matrices (indicating percentage of correct or wrong classification),
- saliency plots (indicating impact of input features to final discriminant),
- intepretability plots based on SHAPley [@NIPS:2017] to evaluate the impact of input features to the discrimination between the classes.

Furthermore, all input features can be plotted with a single command, based on a `yaml` configuration file.
The steps involved in the evaluation stage are illustrated in \autoref{fig:evaluation}. The inference is carried out by running the chosen model on test samples. The evaluation results are rendered in a suite of performance plots.

![Illustration of the evaluation workflow in `Umami`.\label{fig:evaluation}](evaluation.png){ width=60% }


# Conclusions and future work

We present `Umami`, a Python toolkit designed for training machine learning algorithms for jet flavour tagging.
The software is widely used within the ATLAS collaboration to design neural networks which classify jets originating from bottom quarks, charm quarks or other sources.
While the software is customized for this application, it is not limited to it. It is straightforward to modify the expected input features and target classes, such that the general preprocessing and training capabilities can be used in wider contexts. The identification of charged particle tracks or classification of hadronically decaying tau leptons present relevant and adequate possible use-cases.


# Acknowledgements

This work was done as part of the offline software research and development programme of the ATLAS Collaboration, and we thank the collaboration for its support and cooperation.


# References