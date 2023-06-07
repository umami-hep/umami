import pandas
import h5py
import numpy as np

njets = 100

# input
store = pandas.HDFStore("input/train.h5")
jets = store.select("table", stop=njets)


# output
out = h5py.File("test.h5", "w")
out.create_dataset(
    "jets", dtype=np.dtype("f4"), shape=(njets,), compression="lzf"
)
print(jets)

out.close()