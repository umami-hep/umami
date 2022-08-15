# Training the Neural Networks

After the preprocessing is finished, we can start training the different taggers which are supported by the Umami Framework. Theses taggers are DIPS, DIPS Attention, CADS, DL1, different versions of DL1* (like DL1r, DL1d, etc.), Umami, Umami Conditional Attention.
All taggers in the Umami framework are trained using [`tensorflow`](https://www.tensorflow.org/) with the [`Keras`](https://keras.io/) backend. The different models are all able to utilise GPU resources which will drastically shorten the time needed for training. In the following sections, the different steps and the config file are explained in detail which are needed to successfully train one of the taggers.

1. [Start Training your Model](train.md): 
2. [Validate your Training](validate.md): 
3. [Evaluate your Training](evaluate.md):


In general, the train config file consists of 4 parts:

1. Global Settings (Explained in [Start Training your Model](train.md))
2. Network Settings (Explained in [Start Training your Model](train.md))
3. Validation Settings (Explained in [Validate your Training](validate.md))
4. Evaluation Settings (Explained in [Evaluate your Training](evaluate.md))

Example train config files for the different taggers can be found [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/tree/master/examples/training). Using DIPS as an example in the following sections, the different options of the train config are explained. While the basic options needed/provided inside the config files are the same for all taggers, some options are only available for some other.