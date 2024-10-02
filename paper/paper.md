---
title: 'Umami: A Python toolkit for jet flavour tagging'
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
    affiliation: 1
  - name: Joschka Birk
    orcid: 0000-0002-1931-0127
    affiliation: 13
  - name: Maxence Draguet
    orcid: 0000-0003-1530-0519
    affiliation: 3
  - name: Stefano Franchellucci
    orcid: 0000-0003-0695-0798
    affiliation: 4
  - name: Alexander Froch
    orcid: 0000-0002-8259-2622
    affiliation: 4
  - name: Philipp Gadow
    orcid: 0000-0003-4475-6734
    affiliation: 5
  - name: Daniel Hay Guest
    orcid: 0000-0002-4305-2295
    affiliation: 6
  - name: Manuel Guth
    orcid: 0000-0002-6647-1433
    affiliation: 4
  - name: Nicole Michelle Hartman
    orcid: 0000-0001-9111-4916
    affiliation: 7
  - name: Michael Kagan
    orcid: 0000-0002-3386-6869
    affiliation: 8
  - name: Osama Karkout
    orcid: 0000-0002-4907-9499
    affiliation: 9
  - name: Dmitrii Kobylianskii
    orcid: 0009-0002-0070-5900
    affiliation: 10
  - name: Ivan Oleksiyuk
    orcid: 0000-0002-4784-6340
    affiliation: 4
  - name: Nikita Ivvan Pond
    orcid: 0000-0002-5966-0332
    affiliation: 1
  - name: Frederic Renner
    orcid: 0000-0002-9475-3075
    affiliation: 11
  - name: Sebastien Rettie
    orcid: 0000-0002-7092-3893
    affiliation: 5
  - name: Victor Hugo Ruelas Rivera
    orcid: 0000-0002-2116-048X
    affiliation: 6
  - name: Tomke Schröer
    orcid: 0000-0001-7967-6385
    affiliation: 4
  - name: Martino Tanasini
    orcid: 0000-0002-6313-4175
    affiliation: 12
  - name: Samuel Van Stroud
    orcid: 0000-0002-7969-0301
    affiliation: 1
  - name: Janik Von Ahnen
    orcid: 0000-0003-4032-0079
    affiliation: 11
  
  
  
affiliations:
 - name: University College London, United Kingdom
   index: 1
 - name: Albert-Ludwigs-Universität Freiburg, Germany
   index: 2
 - name: University of Oxford, United Kingdom
   index: 3
 - name: Université de Genève, Switzerland
   index: 4
 - name: European Laboratory for Particle Physics CERN, Switzerland
   index: 5
 - name: Humboldt University Berlin, Germany
   index: 6
 - name: Technical University of Munich, Germany
   index: 7
 - name: SLAC National Accelerator Laboratory, United States of America
   index: 8
 - name: Nikhef National Institute for Subatomic Physics and University of Amsterdam, Netherlands
   index: 9
 - name: Department of Particle Physics and Astrophysics, Weizmann Institute of Science, Israel
   index: 10
 - name: Deutsches Elektronen-Synchrotron DESY, Germany
   index: 11
 - name: Stony Brook University, United States of America
   index: 12
 - name: University of Hamburg, Germany
   index: 13


date: 02 October 2024
bibliography: paper.bib

---

# Summary
Flavour-tagging, the identification of collimated sprays of particles ("jets") originating from bottom and charm quarks, is a critically important technique in the data analysis of the ATLAS experiment [@ATLAS:2008] at the Large Hadron Collider [@Evans:2008]. It is applied in precision measurements of the Standard Model of particle physics,
which is the theory describing fundamental forces and classifying all known elementary particles, as well as in searches for yet unknown phenomena.
The long lifetime, high mass, and large decay multiplicity of hadrons containing bottom and charm quarks provide distinct signatures in particle detectors which can be exploited by machine learning algorithms.
Excellent knowledge of the detector and the physics processes at hand enables simulations to provide a high-quality training dataset representative of recorded ATLAS data.
The `Umami` software toolkit provides a unified data pipeline, definition of the algorithms, training and performance evaluation with a high degree of automation.

# Statement of need

`Umami` is a Python toolkit for training and evaluating machine learning algorithms used in high energy physics for jet flavour tagging.
The creation and training of production-grade machine learning models is supported by the `TensorFlow` and `keras` packages. The training datasets feature highly imbalanced distributions among the target classes and input features of vastly different magnitude. Consequentially, the preprocessing of the training data requires resampling to provide balanced datasets and transformation of the input features by scaling and shifting.

`Umami` provides a class-based and user-friendly interface with `yaml` configuration files to steer the data preprocessing and the training of deep neural networks. It is available as a Python application and is also distributed via Linux container images. `Umami` was designed to be used by researchers in the ATLAS collaboration and is open to be applied in a more general context.

# Related work

The application of machine learning in high energy physics, particularly for the classification of jets, is a common and critically important technique [@Guest:2018; @Cagnotta:2022]. In contrast to previous efforts in jet flavour tagging [@Bols:2020; @ATLAS:2019], the current state-of-the-art algorithms [@Qu:2022] rely on specialised toolkits, such as the `Weaver` framework [@Qu:2020]. These toolkits enable the design of algorithms by taking care of input processing, steering the training on large datasets and providing performance metrics. `Umami` provides the required functionality to define, train and evaluate the algorithms used in ATLAS data analysis.

# Software description

The `Umami` toolkit provides an integrated workflow including input data preprocessing, algorithm training, and performance evaluation. 

The algorithms are trained on simulated physics processes which provide jets originating from bottom and charm quarks, as well as the background processes which produce jets originating from other sources, such as light-flavour quarks, gluons, or hadronically decaying tau leptons. The input features to the algorithm provide discrimination between the processes. A more detailled discussion of the input features for jet flavour tagging is provided in Ref. [@ATLAS:2019]. The preprocessing in `Umami` addresses several challenges provided both by the nature of the training datasets and the input features.

Using `Umami` is not limited to jet flavour tagging but provides support for a broad range of applications. The preprocessing capabilities are demonstrated with simulated physics processes from the JetClass dataset [@JetClass:2022] to distinguish jets originating from Higgs boson decays from jets originating from top quark decays. This represents a similar but slightly different use of machine learning algorithms for jet classification. The software is flexible enough to address this task with only minimal modifications in configuration files.

\autoref{fig:eta} shows the absolute value of the pseudorapidity $\eta$ of the jets from Higgs boson decays to b-quarks, Higgs boson decays to c-quarks, and to top quarks before and after the re-sampling step in the preprocessing. The total number of events in each class is equalised and the shape differences between classes are removed by the resampling.

![Distributions of the pseudorapidity $\eta$ of jets from Higgs boson decays to b-quarks ($H \rightarrow b\overline{b}$-jets), Higgs boson decays to c-quarks ($H \rightarrow c\overline{c}$-jets), and to top quarks (Top) before and after resampling.\label{fig:eta}](eta.png){ width=90% }

Different architectures of neural networks, including Deep Multi-Layer-Perceptrons [@LeCun:2015] and Deep Sets, are supported in `Umami` for definition with configuration files.
The training is performed with `keras` using the `TensorFlow` back-end and the Adam optimiser [@Kingma:2015], supporting the use of GPU resources to shorten the required time to train the networks by an order of magnitude.

The performance of the chosen model can be evaluated in publication-grade plots, which are steered with configuration files. The plots are created using the `matplotlib` [@Hunter:2007] and `puma` [@Birk:2023] Python libraries.

# Conclusions and future work

We present `Umami`, a Python toolkit designed for training machine learning algorithms for jet flavour tagging. Its strong point is that it unifies the steps for preprocessing of the training samples, the training and validation of the resulting models in a mostly automated and user-friendly way. The software is widely used within the ATLAS collaboration to design neural networks which classify jets originating from bottom quarks, charm quarks or other sources.
While the software is customised for this application, it is not limited to it. It is straightforward to modify the expected input features and target classes, such that the general preprocessing and training capabilities can be used in wider contexts. The identification of charged particle tracks or classification of hadronically decaying tau leptons present relevant and adequate possible use-cases.


# Acknowledgements

This work was done as part of the offline software research and development programme of the ATLAS Collaboration, and we thank the collaboration for its support and cooperation.


# References