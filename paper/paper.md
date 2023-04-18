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
  - name: Alexander Froch
    orcid: 0000-0002-8259-2622
    equal-contrib: true
    affiliation: 1
  - name: Manuel Guth
    orcid: 0000-0002-6647-1433
    equal-contrib: true
    affiliation: 2
  - name: Joschka Birk
    orcid: 0000-0002-1931-0127
    equal-contrib: true
    affiliation: 1
  - name: Philipp Gadow
    orcid: 0000-0003-4475-6734
    equal-contrib: true
    affiliation: 3
  - name: Maxence Draguet
    equal-contrib: true
    affiliation: 4
  - name: Tomke Schröer
    orcid: 0000-0001-7967-6385
    equal-contrib: true
    affiliation: 2
  - name: Samuel Van Stroud
    orcid: 0000-0002-7969-0301
    equal-contrib: true
    affiliation: 5
  - name: Victor Hugo Ruelas Rivera
    orcid: 0000-0002-2116-048X
    equal-contrib: true
    affiliation: 6
  - name: Frederic Renner
    orcid: 0000-0002-9475-3075
    equal-contrib: true
    affiliation: 3
  - name: Jackson Barr
    orcid: 0000-0002-9752-9204 
    equal-contrib: true
    affiliation: 5
  - name: Janik Von Ahnen
    orcid: 0000-0003-4032-0079
    equal-contrib: true
    affiliation: 3
  - name: Stefano Franchellucci
    orcid: 0000-0003-0695-0798
    equal-contrib: true
    affiliation: 2
  - name: Sebastien Rettie
    orcid: 0000-0002-7092-3893
    equal-contrib: true
    affiliation: 7
  - name: Nikita Ivvan Pond
    orcid: 0000-0002-5966-0332
    equal-contrib: true
    affiliation: 5
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
date: 01 May 2023
bibliography: paper.bib

---

# Summary

Flavour-tagging, the identification of jets originating from bottom and charm quarks, is a critically important technique in the data analysis of the ATLAS experiment [@ATLAS:2008] at the Large Hadron Collider [@Evans:2008]. It is applied in precision measurements of the Standard Model and the Higgs boson, as well as in searches for yet unknown phenomena.
The long lifetime, high mass, and large decay multiplicity of hadrons containing bottom and charm quarks provide distinct signatures of charged particle trajectories in the detector which can be exploited by machine learning algorithms.
The excellent knowledge of the detector and the physics processes at hand enables simulations to provide high-quality training data for algorithms.
The `Umami` software toolkit provides a unified data pipeline, definition of the algorithms, training and performance evaluation with a high degree of automation.

# Statement of need

`Umami` is a Python toolkit for training and evaluating machine learning algorithms used in high-energy-physics jet flavour tagging.
The creation and training of production-grade machine learning models is supported by the TensorFlow [@tensorflow:2015] and Keras [@chollet:2015] packages for Python. The training datasets feature highly imbalanced distributions among the target classes and input features of vastly different magnitude. Consequentially, the preprocessing of the training data requires resampling to provide balanced datasets and normalisation of the input features by scaling and shifting.

`Umami` provides a class-based and user-friendly interface with configuration files to steer the data preprocessing and the training of deep neural networks. `Umami` was designed to be used by researchers in the ATLAS collaboration and is open to be applied in more general context.

# Related work

The application of machine learning in high energy physics, particularly for the classification of jets, is a common and critically important technique [@Guest:2018; @Cagnotta:2022]. In contrast to previous efforts in jet flavour tagging [@Bols:2020; @ATLAS:2019], the current state-of-the-art algorithms [@Qu:2020;Qu:2022] rely on specialised toolkits, such as the `Weaver` research and development framework. These toolkits enable the design of algorithms by taking care of input processing, steering the training on large datasets and providing performance metrics with fast inference.

# Development Notes

The development of the package is based on PEP8 standards. 
They are enforced by a continuous integration pipeline in a GitLab project, using a linter and the `black` command-line tool. The code quality is tested as part of the continuous integration pipeline with the `pytest` module, using unit tests and integration tests.
The package has been released as open-source software under the Apache v2 license. 

# Software description

The `Umami` toolkit provides an integrated workflow including input data preprocessing, algorithm training, and performance evaluation.

## Preprocessing

The algoriths are trained on simulated physics processes which provide jets originating from bottom quarks, as well as background processes. Several datasets can be combined to a hybrid sample, which is populated over a large momentum range.
The classes in the input dataset are highly unbalanced. Consequentially, `Umami` provides under- and oversampling methods to ensure similar kinematic distributions for the jets of different target classes.
The range of values of the input features on which the algorithm is trained can differ considerably. Consequentially, `Umami` normalises the range of the independent variables and creates a `json` file with scaling and shifting parameters.
The resulting training data has balanced target classes and normalised input features. It can be stored in the binary `TFRecords` format to improve reading speed.
The steps involved in the preprocessing workflow are illustrated in Fig. \autoref{fig:preprocessing}.

![Illustration of the preprocessing workflow in `Umami`.\label{fig:preprocessing}](preprocessing.png){ width=20% }

## Training

Different architectures of neural networks are supported in `Umami` for definition with configuration files.
The training is performed with `TensorFlow` using the `keras` back-end and the Adam optimizer[@Kingma:2015], supporting the use of GPU resources to drastically shorten the required time. 
Parameters defined in the configuration file include the batch size, the number of epochs, as well as the learning rate.
The resulting model from each epoch during the training is saved. These models are evaluated on a validation dataset to identify the optimal configuration. Typical performance metrics include the validation loss and the efficiency in identifying the correct jet labels. These can be plotted as a function of the training epoch to select the epoch which corresponds to the optimal performance of the trained model.
The steps involved in the training workflow are illustrated in Fig. \autoref{fig:training}.

![Illustration of the training workflow in `Umami`.\label{fig:training}](training.png){ width=20% }


## Performance evaluation


The performance of the chosen model can be evaluated in publication-grade plots, which are steered with configuration files.
Typical performance plots include

- receiver-operator-characteristics (ROC),
- efficiency of the signal class and rejection of background classes as functions of certain variables,
- confusion matrices (indicating percentage of correct or wrong classification)
- saliency maps based on SHAPley [@NIPS:2017] to evaluate the impact of input features to the discrimination between the classes

Furthermore, all input features can be plotted with a single command.

# Conclusions and future work

We present `Umami`, a python toolkit designed for training machine learning algorithms for jet flavour tagging.
The software is widely used within the ATLAS collaboration to design neural networks which classify jets.
While the software is customized for this application, it is not limited to it. It is straightforward to modify the expected input features and target classes, such that the general preprocessing and training capabilities can be used in wider contexts. The identification of charged particle tracks or classification of hadronically decaying tau leptons present relevant and adequate possible use-cases.


# Acknowledgements

We acknowledge the support of 
ANPCyT, Argentina; 
YerPhI, Armenia; 
ARC, Australia; 
BMWFW and FWF, Austria; 
ANAS, Azerbaijan; 
CNPq and FAPESP, Brazil; 
NSERC, NRC and CFI, Canada; 
CERN; 
ANID, Chile; 
CAS, MOST and NSFC, China; 
Minciencias, Colombia; 
MEYS CR, Czech Republic; 
DNRF and DNSRC, Denmark; 
IN2P3-CNRS and CEA-DRF/IRFU, France; 
SRNSFG, Georgia; 
BMBF, HGF and MPG, Germany; 
GSRI, Greece; 
RGC and Hong Kong SAR, China; 
ISF and Benoziyo Center, Israel; 
INFN, Italy; 
MEXT and JSPS, Japan; 
CNRST, Morocco; 
NWO, Netherlands; 
RCN, Norway; 
MEiN, Poland; 
FCT, Portugal; 
MNE/IFA, Romania; 
MESTD, Serbia; 
MSSR, Slovakia; 
ARRS and MIZ\v{S}, Slovenia; 
DSI/NRF, South Africa; 
MICINN, Spain; 
SRC and Wallenberg Foundation, Sweden; 
SERI, SNSF and Cantons of Bern and Geneva, Switzerland; 
MOST, Taiwan; 
TENMAK, T\"urkiye; 
STFC, United Kingdom; 
DOE and NSF, United States of America. 
In addition, individual groups and members have received support from 
BCKDF, CANARIE, Compute Canada and CRC, Canada; 
PRIMUS 21/SCI/017 and UNCE SCI/013, Czech Republic; 
COST, ERC, ERDF, Horizon 2020 and Marie Sk{\l}odowska-Curie Actions, European Union; 
Investissements d'Avenir Labex, Investissements d'Avenir Idex and ANR, France; 
DFG and AvH Foundation, Germany; 
Herakleitos, Thales and Aristeia programmes co-financed by EU-ESF and the Greek NSRF, Greece; 
BSF-NSF and MINERVA, Israel; 
Norwegian Financial Mechanism 2014-2021, Norway; 
NCN and NAWA, Poland; 
La Caixa Banking Foundation, CERCA Programme Generalitat de Catalunya and PROMETEO and GenT Programmes Generalitat Valenciana, Spain; 
G\"{o}ran Gustafssons Stiftelse, Sweden; 
The Royal Society and Leverhulme Trust, United Kingdom.


# References