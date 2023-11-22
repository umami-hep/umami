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
dtype = np.dtype(
    [
        ("pt", np.float16),
        ("eta", np.float16),
        ("mass", np.float16),
        ("R10TruthLabel_R22v1", np.int32),
    ]
)

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
