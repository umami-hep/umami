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
    affiliation: 2
  - name: Maxence Draguet
    affiliation: 3
  - name: Stefano Franchellucci
    orcid: 0000-0003-0695-0798
    affiliation: 4
  - name: Alexander Froch
    orcid: 0000-0002-8259-2622
    affiliation: 2
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
 - name: INFN Genova and Universita' di Genova, Italy
   index: 12



date: 09 August 2023
bibliography: paper.bib

---

# Summary

Flavour-tagging, the identification of jets originating from bottom and charm quarks, is a critically important technique in the data analysis of the ATLAS experiment [@ATLAS:2008] at the Large Hadron Collider [@Evans:2008]. It is applied in precision measurements of the Standard Model, e.g. in characterisations of the Higgs boson properties, as well as in searches for yet unknown phenomena.
The long lifetime, high mass, and large decay multiplicity of hadrons containing bottom and charm quarks provide distinct signatures in particle detectors which can be exploited by machine learning algorithms.
Excellent knowledge of the detector and the physics processes at hand enables simulations to provide a high-quality training dataset representative of recorded ATLAS data.
The `Umami` software toolkit provides a unified data pipeline, definition of the algorithms, training and performance evaluation with a high degree of automation.

# Statement of need

`Umami` is a Python [@Rossum:2009] toolkit for training and evaluating machine learning algorithms used in high energy physics for jet flavour tagging.
The creation and training of production-grade machine learning models is supported by the `TensorFlow` [@tensorflow:2015] and `keras` [@chollet:2015] packages. The training datasets feature highly imbalanced distributions among the target classes and input features of vastly different magnitude. Consequentially, the preprocessing of the training data requires resampling to provide balanced datasets and transformation of the input features by scaling and shifting.

`Umami` provides a class-based and user-friendly interface with `yaml` [@YAML:2021] configuration files to steer the data preprocessing and the training of deep neural networks. It is deployed as a Python module which can be installed with `setuptools` [@setuptools:2023] or used via Docker images [@Merkel:2014]. `Umami` was designed to be used by researchers in the ATLAS collaboration and is open to be applied in a more general context.

# Related work

The application of machine learning in high energy physics, particularly for the classification of jets, is a common and critically important technique [@Guest:2018; @Cagnotta:2022]. In contrast to previous efforts in jet flavour tagging [@Bols:2020; @ATLAS:2019], the current state-of-the-art algorithms [@Qu:2022] rely on specialised toolkits, such as the `Weaver` framework [@Qu:2020]. These toolkits enable the design of algorithms by taking care of input processing, steering the training on large datasets and providing performance metrics. `Umami` provides the required functionality to define, train and evaluate the algorithms used in ATLAS data analysis.


# Development Notes

The development of the package adheres to PEP8 standards [@PEP8:2001]. They are enforced by a continuous integration pipeline in a GitLab project, using the `flake8` linter [@flake8:2023] and the `black` command-line tool for code formatting [@black:2023]. The code quality is tested as part of the continuous integration pipeline with the `pytest` module [@pytest:2004], using unit tests and integration tests.
Documentation of the software is built automatically with the `mkdocs` [@mkdocs:2023] and `sphinx` [@sphinx:2023] modules and deployed to the website [`https://umami.docs.cern.ch`](https://umami.docs.cern.ch).
The `Umami` toolkit has been released as open-source software under the Apache v2 license. 

# Software description

The `Umami` toolkit provides an integrated workflow including input data preprocessing, algorithm training, and performance evaluation. Furthermore, it interfaces to `lwtnn` [@Guest:2022] to export the trained models to `json` files for `C++` deployment in the ATLAS software stack [@ATLAS:2021].

## Preprocessing

The algorithms are trained on simulated physics processes which provide jets originating from bottom and charm quarks, as well as the background processes which produce jets originating from other sources, such as light-flavour quarks, gluons, or hadronically decaying tau leptons. The input features to the algorithm provide discrimination between the processes. A more detailled discussion of the input features for jet flavour tagging is provided in Ref. [@ATLAS:2019]. The preprocessing in `Umami` addresses several challenges provided both by the nature of the training datasets and the input features.

The steps involved in the preprocessing workflow are illustrated in \autoref{fig:preprocessing}.

![Illustration of the preprocessing workflow in `Umami`. Input files with simulated physics processes undergo several stages to provide a training sample, as well as a `json` file with scaling and shifting information ("Scale Dict") and validation/testing samples. These stages include a Preparation stage to define the target classes for the training, a Resampling stage to balance the classes, a Scaling/Shifting stage to transform the range of variables used for training, and a Writing stage to output the resulting samples. The validation/testing samples can also undergo the same resampling as the training sample (not shown).\label{fig:preprocessing}](preprocessing.png){ width=60% }

Typically, the three classes "b-jets" (originating from bottom quarks), "c-jets" (originating from charm quarks), and "light-flavour jets" (originating from gluons and light-flavour quarks) are considered. Several datasets with different physics processes can be combined to a hybrid sample, which is populated over a large jet momentum range.
The classes in the input dataset are highly imbalanced because the physics processes will predominantly produce light-flavoured jets instead of b-jets or c-jets. 
Equalising also the distributions for all classes for a certain set of features allows for the parameterisation of the algorithm performance in terms of these features, which can be desirable.
Consequentially, `Umami` provides under- and oversampling methods as well as a weighting method to ensure similar kinematic distributions for the jets of different target classes. 

In the first step "Preparation", datasets which are pure in the target classes are extracted from the simulated physics processes. Then, the training datasets are resampled in the "Resampling" step to provide balanced distributions between classes.

The range of values of the input features on which the algorithm is trained can differ considerably. Consequentially, `Umami` transforms the range of the variables used in training and creates a `json` file with scaling and shifting parameters.
The input features are scaled and shifted in the "Scaling/Shifting" step.
The resulting training data has balanced target classes and transformed input features. 
Finally, the training sample is written to disk, together with the `json` file and datasets for validation and performance evaluation. The resulting datasets can be stored either as an `hdf5` file [@hdf5:2023] or in the binary `TFRecords` format to improve reading speed provided by `TensorFlow`.
The validation and testing samples can undergo the same resampling procedure as the training data if desired by the user.

Using `Umami` is not limited to jet flavour tagging but provides support for a broad range of applications. The preprocessing capabilities are demonstrated with simulated physics processes from the JetClass dataset [@JetClass:2022] to distinguish jets originating from Higgs boson decays from jets originating from top quark decays. This represents a similar but slightly different use of machine learning algorithms for jet classification. The software is flexible enough to address this task with only minimal modifications in configuration files. A comprehensive discussion of flavour tagging algorithm training is provided in Ref. [@ATLAS:2019].

\autoref{fig:eta} shows the pseudorapidity $\eta$ and its absolute value $|\eta|$ of the jets from Higgs boson decays to b-quarks, Higgs boson decays to c-quarks, and to top quarks before and after the re-sampling step in the preprocessing. The total number of events in each class is equalised and the shape differences between classes are removed by the resampling.

![Distributions of the pseudorapidity $\eta$ of jets from Higgs boson decays to b-quarks ($H \rightarrow b\overline{b}$-jets), Higgs boson decays to c-quarks ($H \rightarrow c\overline{c}$-jets), and to top quarks (Top) before and after resampling.\label{fig:eta}](eta.png){ width=90% }

Similarly, \autoref{fig:mass} shows the invariant mass before and after pre-processing, including the transformation of the dimensional quantity to a smaller range centered around zero. She scaling and shifting results in input features which are centred around zero and have the distribution in similar order of magnitude as other features.

![Distributions of the invariant mass jets from Higgs boson decays to b-quarks ($H \rightarrow b\overline{b}$-jets), Higgs boson decays to c-quarks ($H \rightarrow c\overline{c}$-jets), and to top quarks (Top) before and after pre-processing.\label{fig:mass}](mass.png){ width=90% }


## Training

Different architectures of neural networks, including Deep Multi-Layer-Perceptrons [@LeCun:2015] and Deep Sets [@Zaheer:2017], are supported in `Umami` for definition with configuration files.
The training is performed with `keras` using the `TensorFlow` back-end and the Adam optimiser [@Kingma:2015], supporting the use of GPU resources to shorten the required time to train the networks by an order of magnitude.

The steps involved in the training workflow are illustrated in \autoref{fig:training}. After the "Training" step, the optimal model configuration is chosen in the "Validation" step by evaluating the trained model with the `json` file providing the scaling and shifting parameters on the validation sample which was prepared in the preprocessing.

![Illustration of the training workflow in `Umami`. The training sample is processed to determine the optimal weights of the network with a given loss function. Using the validation sample and applying to it the scaling and shifting parameters from the `json` file  ("Scale Dict") obtained from the preprocessing, the performance of the training is validated to chose a certain model.\label{fig:training}](training.png){ width=60% }

The resulting model from each epoch (in which the whole dataset was processed by the algorithm) is saved during the training. These models are evaluated on a validation dataset to identify the network weights corresponding to the epoch with the best performance. Typical performance metrics include the validation loss and the efficiency in identifying the correct jet labels. These can be plotted as a function of the training epoch to guide the selection.

As an example, a deep neural network is trained on the previously discussed JetClass dataset to separate jets originating from Higgs boson decays from jets originating from top quark decays. \autoref{fig:loss} shows the loss function which is minimised while training the model on the training sample (purple curve) as a function of the epoch together with the loss function evaluated on the validation sample (green curve). Similarly, \autoref{fig:accuracy} shows the accuracy.

![The loss function which is minimised while training a deep neural network for separating jets originating from Higgs boson decays from jets originating from top quark decays in the JetClass dataset, shown both for the training data (purple curve) and the validation data (green curve).\label{fig:loss}](loss-plot.png){ width=80% }

![The accuracy for classifying jets originating from Higgs boson decays while training a deep neural network for separating jets originating from Higgs boson decays from jets originating from top quark decays in the JetClass dataset, shown both for the training data (purple curve) and the validation data (green curve).\label{fig:accuracy}](accuracy-plot.png){ width=80% }

Typically, in the early epochs the accuracy on the training data is higher than on the validation data which are not used in training. Convergence of the two curves for later epochs demonstrates that the model does generalise well and does not pick up peculiarities of the training data ("overtraining").


## Performance evaluation


The performance of the chosen model can be evaluated in publication-grade plots, which are steered with configuration files. The plots are created using the `matplotlib` [@Hunter:2007] and `puma` [@Birk:2023] Python modules.
Typical performance plots include

- receiver-operator-characteristics (ROC),
- efficiency of the signal class and rejection of background classes as functions of certain variables,
- confusion matrices (indicating percentage of correct or wrong classification),
- interpretability plots based on SHAPley [@NIPS:2017] to evaluate the impact of input features to the discrimination between the classes, as well as saliency plots indicating impact of input features to final discriminant

Furthermore, all input features can be plotted with a single command, based on a `yaml` configuration file.
The steps involved in the evaluation stage are illustrated in \autoref{fig:evaluation}. The inference is carried out by running the chosen model on test samples. The evaluation results are rendered in a suite of performance plots.

![Illustration of the evaluation workflow in `Umami`. The chosen model after training is evaluated on the testing sample with the scaling and shifting parameters applied to it from the `json` file  ("Scale Dict") obtained from the preprocessing. The results of the evaluation can be plotted.\label{fig:evaluation}](evaluation.png){ width=60% }

The plotting capabilities of the `Umami` toolkit are used to evaluate the performance of flavour tagging algorithms used in ATLAS and in the corresponding publications. \autoref{fig:evaluation} [@ATLAS:2019] shows the light-flavour jet and c-jet rejection factors as a function of the b-jet efficiency for three different ATLAS flavour tagging algorithms MV2c10, DL1, and DL1r.

![The light-flavour jet and c-jet rejection factors as a function of the b-jet efficiency for the high-level b-tagging algorithms MV2c10, DL1, and DL1r. The lower two panels show the ratio of the light-flavour jet rejection and the (c)-jet rejection of the algorithms to MV2c10. The statistical uncertainties of the rejection factors are calculated using binomial uncertainties and are indicated as coloured bands. Reproduced from [@ATLAS:2019].\label{fig:dl1r}](fig_09.png){ width=80% }



# Conclusions and future work

We present `Umami`, a Python toolkit designed for training machine learning algorithms for jet flavour tagging. Its strong point is that it unifies the steps for preprocessing of the training samples, the training and validation of the resulting models in a mostly automated and user-friendly way. The software is widely used within the ATLAS collaboration to design neural networks which classify jets originating from bottom quarks, charm quarks or other sources.
While the software is customised for this application, it is not limited to it. It is straightforward to modify the expected input features and target classes, such that the general preprocessing and training capabilities can be used in wider contexts. The identification of charged particle tracks or classification of hadronically decaying tau leptons present relevant and adequate possible use-cases.


# Acknowledgements

This work was done as part of the offline software research and development programme of the ATLAS Collaboration, and we thank the collaboration for its support and cooperation.


# References