# Instructions to train the Umami-tagger with the umami framework


## Preprocessing

Same as [Dl1r](DL1r-instructions.md) except for also writing the track information to the samples. Note that tau jets are not yet supported in Umami.

## Training

After all the files are ready we can start with the training. The config file for the Umami training is [umami-PFlow-Training-config.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/umami-PFlow-Training-config.yaml).

It contains the information about the neural network architecture as well as about the files for training, validation and testing.

To run the training, use the following command

```bash
train_umami.py -c examples/umami-PFlow-Training-config.yaml
```

Alternatively, if you are working out of the DESY Zeuthen servers, `warp.zeuthen.desy.de`, you can train using the batch system via `qsub` and GPU support by giving it the `zeuthen` flag

```bash
train_umami.py -c examples/umami-PFlow-Training-config.yaml --zeuthen
```

The job will output a log to the current working directory and copy the results to the current working directory when it's done. The options for the job (time, memory, space, etc.) can be changed in `umami/institutes/zeuthen/train_job.sh`.

## Plotting

### Rejection Rates for Validation Sample

During the training the important metrics are saved in a json file and from this file the rejection rates per epoch can be plotted:

```bash
plotting_epoch_performance.py -c examples/umami-PFlow-Training-config.yaml -d umami_dev/validation_WP0p77_fc0p018_100000jets_Dict.json
```

If you want to re-evaluate the training with the valdiation samples but different charm fraction, working point or number of jets then you can do so by omitting the `-d` option like this:

```bash
plotting_epoch_performance.py -c examples/umami-PFlow-Training-config.yaml --beff 0.6 --cfrac 0.08 --nJets 300000
```

### Detailed Performance Evaluation on Test Sample

Finally we can evaluate our model.

You first need to choose which epoch you want to use for the evaluation (easiest by looking at the performance plots vs. epochs) and then run

```bash
evaluate_model.py -c examples/umami-PFlow-Training-config.yaml -e 230
```

Next you need to adapt the plotting config file [examples/plotting_umami_config_Umami.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/plotting_umami_config_Umami.yaml) providing the epoch and model name in the `Eval_parameters`. The plots can then be created by running the following command

```bash
plotting_umami.py -c examples/plotting_umami_config_Umami.yaml
```
