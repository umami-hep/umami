## Writing Train Sample
In the final step of the preprocessing, the resampled, scaled and shifted training jets are written to disk in a format, that can be used for training. For this, the collection of the training sample will get different names and data types. The collections are replaced with datasets with unstructured `numpy.ndarray`s. The names/shapes of these new datasets in the final training file can be found in the table below:

| Before writing | After writing | Shape | Comment |
| -------------- | ------------- | ----- | ------- |
| `jets` | `X_train` | `(n_jets, n_jet_variables)` | |
| `<tracks_name>` | `X_<tracks_name>_train` | `(n_jets, n_tracks, n_track_variables)` | `<tracks_name>` is the name of the track collection in the `.h5` files coming from the training-dataset-dumper. |
| `labels` | `Y_train` | `(n_jets, n_jet_classes)` | One-hot encoded truth labels. The `n_jet_classes` are the `class_labels` defined in the preprocessing config. The value `0` here corresponds to the jet origin which is on index `0` in the `class_labels` list. |
| `labels` | `flavour` | `(n_jets,)` | Sparse representation of the jet labels. The value `0` here corresponds to the jet origin which is on index `0` in the `class_labels` list. |
| `<tracks_name>_labels` | `Y_<tracks_name>_train` | `(n_jets, n_tracks, n_track_truth_variables)` | This is the sparse representation of the `track_truth_variables`. |

In the final training file, the column information (and therefore which column corresponds to which variable) is not longer available. You can run `h5ls -v` on the file to get some information about the variables for each of the datasets. The variables for the specific jet and track(s) datasets will be shown as a attribute of the dataset. The order of this variables shown is also the order of the variable columns in the dataset.

### Config File
For the final writing step, only a few options are needed. Those are shown/explained below

```yaml
§§§examples/PFlow-Preprocessing.yaml:272:281§§§
```

| Setting | Explanation |
| ------- | ----------- |
| `compression` | Decide, which compression is used for the final training sample. Due to slow loading times, this should be `null`. Possible options are for example `gzip`. |
| `precision` | The precision of the final output file. The values are saved with the given precision to save space. |
| `convert_to_tfrecord` | Options for the conversion to tfrecords. Possible options to define are the `chunk_size` which gives the number of samples saved per file and the number of additional variables to be saved in tf records `N_Add_Vars`. |

When you want to train with conditional information, i.e. jet $p_T$ and $\eta$, the corresponding model (CADS) will load the jet information directly from the train file when using `.h5`. When you want to use `TFRecords`, you need to define the amount of variables that are added to extra to the files with `N_Add_Vars`. Until now, when using `2`, this will use the first two available jet variables, which are by default jet $p_T$ and $\eta$.

### Running the Writing Step
The writing of the final training sample can be started via the command

```bash
preprocessing.py --config <path to config file> --write
```

The writing step will take some time, depending on the amount of jets you want to use for training and also if you are using track collections.

#### TFRecords writing
If you are saving the tracks it might be useful to save your samples as a directory with [tf Records](https://www.tensorflow.org/tutorials/load_data/tfrecord). This can be done by using `--to_records` instead of `--write`.
Important: you need to have ran `--write` beforehand.

```bash
preprocessing.py --config <path to config file> --to_records
```

??? info "TF records"

    TF records are the Tensorflow's own file format to store datasets. Especially when working with large datasets this format can be useful. In TF records the data is saved as a sequence of binary strings. This has the advantage that reading the data is significantly faster than from a .h5 file. In addition the data can be saved in multiple files instead of one big file containing all data. This way the reading procedure can be parallelized which speeds up the whole training.
    Besides of this, since TF records are the Tensorflow's own file format, it is optimised for the usage with Tensorflow. For example, the dataset is not stored completely in memory but automatically loaded in batches as soon as needed.