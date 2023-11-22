In the preparation and preprocessing section of this tutorial, we focus on converting and preparing the data for our machine learning model. This step involves organizing and cleaning the data, as well as applying suitable preprocessing techniques to ensure that the data is in the optimal format for effective model training.

## Prerequisites

**Step 1** Download the JetClass dataset. For the purpose of this tutorial, we only use a subset of the dataset.

```bash
mkdir -p data/jetclass/train_10M && cd data/jetclass/train_10M
wget https://zenodo.org/records/6619768/files/JetClass_Pythia_train_100M_part0.tar
tar xvf JetClass_Pythia_train_100M_part0.tar
```


**Step 2** Convert the JetClass dataset to the Umami hdf5 format. We will use the `R10TruthLabel_R22v1` variable to encode the different target classes. Note that for the following script it is assumed that you have installed the following python packages.

```
pip install uproot h5py numpy lz4 xxhash
```

You can create a script `convert_to_umami` with the following content and execute it to create the input file for Umami.

??? example "`convert_to_umami.py` script"

    ```python
    import uproot
    import h5py
    import numpy as np
    import os
    import fnmatch

    # Directory containing the ROOT files
    root_dir = "data/jetclass/train_10M"
    output_dir = "data/jetclass/ntuples"
    output_file = "combined_data.h5"

    # File patterns and their corresponding class labels
    file_patterns = {
        "HToBB_*.root": 11,
        "HToCC_*.root": 12,
        "TTBar_*.root": 1,
    }

    # Define the dtype for the structured array
    dtype = np.dtype([
        ("pt", np.float16),
        ("eta", np.float16),
        ("mass", np.float16),
        ("R10TruthLabel_R22v1", np.int32)
    ])

    # List to store data
    all_data = []

    # Process each file pattern
    for pattern, class_label in file_patterns.items():
        for filename in os.listdir(root_dir):
            if fnmatch.fnmatch(filename, pattern):
                file_path = os.path.join(root_dir, filename)

                with uproot.open(file_path) as file:
                    tree = file["tree"]

                    # Extract relevant branches
                    pt = tree["jet_pt"].array(library="np").astype(np.float16)
                    eta = tree["jet_eta"].array(library="np").astype(np.float16)
                    mass = tree["jet_sdmass"].array(library="np").astype(np.float16)

                    # Create structured array
                    data = np.empty(len(pt), dtype=dtype)
                    data["pt"] = pt
                    data["eta"] = eta
                    data["mass"] = mass
                    data["R10TruthLabel_R22v1"] = class_label

                    all_data.append(data)

    # Combine all data
    combined_data = np.concatenate(all_data)

    # Shuffle the combined data
    np.random.shuffle(combined_data)

    # Write data to HDF5 file
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    with h5py.File(output_path, "w") as h5file:
        h5file.create_dataset("jets", data=combined_data)

    print(f"Data written to {output_path}")
    ```

As a result, you should have now a directory `data/jetclass/` which contains the converted dataset as a h5 file in `data/jetclass/ntuples/combined_data.h5`.


## Preprocessing

The preprocessing is steered via configuration files. The relevant files are located in `examples/tutorial_jetclass/`. Please have a look at the files

- `Preprocessing-config.yaml`: main preprocessing files
- `Preprocessing-parameters.yaml`: custom parameters specific to your computing site (you might need to modify this if you deviated from the previous instructions)
- `Preprocessing-samples.yaml`: samples which correspond to the classes in the training
- `Preprocessing-cut_parameters.yaml`: can contain commonly used definitions for selection cuts applied to datasets (not used in this tutorial)

**Step 1** Run the preparation step. This step is needed to preprare the different selected samples.

```bash
preprocessing.py --config examples/tutorial_jetclass/Preprocessing-config.yaml --prepare
```
As a result, you should now see two new directories `data/jetclass/preprocessed/` and `data/jetclass/hybrids/` which contain the output of the preparation step.

**Step 2** Run the resampling step. This step uses `count`-based undersampling to achieve balanced training classes. You will create both a training sample and a hybrid validation sample.

```bash
preprocessing.py --config examples/tutorial_jetclass/Preprocessing-config.yaml --resampling
preprocessing.py --config examples/tutorial_jetclass/Preprocessing-config.yaml --resampling --hybrid_validation
```

**Step 3** Run the scale + shift step. This step processes the training variables to normalise the range of the independent variables.

```bash
preprocessing.py --config examples/tutorial_jetclass/Preprocessing-config.yaml --scaling
```

**Step 4** Run the writing step. This step writes the training dataset to disk. You will also write the validation sample to disk.

```bash
preprocessing.py --config examples/tutorial_jetclass/Preprocessing-config.yaml --write
preprocessing.py --config examples/tutorial_jetclass/Preprocessing-config.yaml --write --hybrid_validation
```


You now have finished all steps of the preprocessing stage. The result is provided in `data/jetclass/preprocessed/`. Have a look at the plots in `data/jetclass/preprocessed/plots`
In the next part you will run a training of a fully connected deep neural network to classify the jets.