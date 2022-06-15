# Instructions to train a Graph-Neural-Network tagger with the umami framework

The following instructions are meant to give a guidline how to train and evaluate the Graph-Neural-Network (GNN) tagger. It is focused on the PFlow training. The repository for the tagger is [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/GNNJetTagger), and dedicated docs are available [here](https://ftag-gnn.docs.cern.ch/)

Further information on the GNN tagger is provided in the algorithms documentation [here](https://ftag-docs.docs.cern.ch/algorithms/GNN/) (access restricted to members of the ATLAS collaboration).

## Sample Preparation

The first step is to obtain the samples for the training. All the samples are listed in [MC-Samples.md](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/MC-Samples.md). For the PFlow training only the ttbar and extended Z' samples from 2017 data taking period (MC16d) were used.

The training ntuples are produced using the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper) which dumps the jets from the PHYSVAL derivations directly into hdf5 files. The processed ntuples are also listed in the table in [MC-Samples.md](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/MC-Samples.md) which can be used for training. If you want to dump your own samples, you should make sure you have the information used in the [GNN config](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/-/blob/r22/configs/single-b-tag/EMPFlowGNN.json).

### Ntuple preparation

After the previous step the ntuples need to be further processed. We can use different resampling approaches to achieve the same pt and eta distribution for all of the used flavour categories.

This processing can be done using the preprocessing capabilities of Umami via the [`preprocessing.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/preprocessing.py) script.

Please refer to the [documentation on preprocessing](https://umami-docs.web.cern.ch/preprocessing/preprocessing/) for additional information.

For the GNN, we use the `PFlow-Preprocessing-GNN.yaml` config file, found [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami-config-tags/-/blob/master/offline/PFlow-Preprocessing-GNN.yaml).

## Training & Evaluation

Please take a look at the [GNN docs](https://ftag-gnn.docs.cern.ch/).
