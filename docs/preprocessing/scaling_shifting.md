## Scaling and Shifting
After the jets and resampled and the training set creation is completed, we need to scale and shift the training jet variables to normalise the range of the independent variables.
[Wikipedia](https://en.wikipedia.org/wiki/Feature_scaling) gives a motivation for the scaling + shifting step:

> Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization. For example, many classifiers calculate the distance between two points by the Euclidean distance. If one of the features has a broad range of values, the distance will be governed by this particular feature. Therefore, the range of all features should be normalized so that each feature contributes approximately proportionately to the final distance. Another reason why feature scaling is applied is that gradient descent converges much faster with feature scaling than without it.

## Run the Scaling and Shifting
The scaling and shifting step calculates the scaling/shifting values for the variables which are defined in the `var_dict` and writes them into the `.json` scaling dict file using the resampled jets from the resamling step. The path for the scaling dict file is given in the preprocessing config option `dict_file` [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/PFlow-Preprocessing.yaml#L270). By default, the path is taken from the [`Preprocessing-parameters.yaml`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/Preprocessing-parameters.yaml) file.
This is invoked with the following command

```bash
preprocessing.py --config <path to config file> --scaling
```
