# Plotting with the Python API

Currently the plotting part of umami is rewritten for better usage with the python api.
For now, the plotting of roc curves was rewritten.

An example of the roc plotting can be found [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/plot_rocs.py).


## ROC curve plotting API
In the following a small example how to plot a roc curve with the umami python api.

first you need to import some packages
```py
import h5py
import numpy as np
import pandas as pd


from umami.metrics import calc_rej
from umami.plotting import roc, roc_plot
```

??? example "Reading `.h5` file"

    ```py
    # this is just an example to read in your h5 file
    # if you have tagger predictions you can plug them in directly in the `disc_fct` as well
    # taking one random ttbar file
    ttbar_file = (
        "user.alfroch.410470.btagTraining.e6337_s3681_r13144_p4931.EMPFlowAll."
        "2022-02-07-T174158_output.h5/user.alfroch.28040424._001207.output.h5"
    )
    with h5py.File(ttbar_file, "r") as f:
    df = pd.DataFrame(
        f["jets"].fields(
            [
                "rnnip_pu",
                "rnnip_pc",
                "rnnip_pb",
                "dipsLoose20210729_pu",
                "dipsLoose20210729_pc",
                "dipsLoose20210729_pb",
                "HadronConeExclTruthLabelID",
            ]
        )[:300000]
    )
    n_test = len(df)
    ```

??? example "Calculating discriminant"

    ```py
    # define a small function to calculate discriminant
    def disc_fct(a: np.ndarray):
        """Tagger discriminant

        Parameters
        ----------
        a : numpy.ndarray
            array with with shape (, 3)
        """
        # you can adapt this for your needs
        return np.log(a[2] / (0.018 * a[1] + 0.982 * a[0]))


    # you can also use a lambda function
    # discs_rnnip = np.apply_along_axis(
    #     lambda a: np.log(a[2] / (0.018 * a[1] + 0.982 * a[0])),
    #     1,
    #     df[["rnnip_pu", "rnnip_pc", "rnnip_pb"]].values,
    # )

    # calculate discriminant
    discs_rnnip = np.apply_along_axis(
        disc_fct, 1, df[["rnnip_pu", "rnnip_pc", "rnnip_pb"]].values
    )
    discs_dips = np.apply_along_axis(
        disc_fct,
        1,
        df[["dipsLoose20210729_pu", "dipsLoose20210729_pc", "dipsLoose20210729_pb"]].values,
    )
    ```

To calculate the rejection values you can do the following or using a results file from umami directly.

??? example "Rejection calculation"

    ```py
    # defining target efficiency
    sig_eff = np.linspace(0.49, 1, 20)
    # defining boolean arrays to select the different flavour classes
    is_light = df["HadronConeExclTruthLabelID"] == 0
    is_c = df["HadronConeExclTruthLabelID"] == 4
    is_b = df["HadronConeExclTruthLabelID"] == 5

    rnnip_ujets_rej = calc_rej(discs_rnnip[is_b], discs_rnnip[is_light], sig_eff)
    rnnip_cjets_rej = calc_rej(discs_rnnip[is_b], discs_rnnip[is_c], sig_eff)
    dips_ujets_rej = calc_rej(discs_dips[is_b], discs_dips[is_light], sig_eff)
    dips_cjets_rej = calc_rej(discs_dips[is_b], discs_dips[is_c], sig_eff)
    ```

??? example "Reading in results"

    ```py
    # Alternatively you can simply use a results file with the rejection values
    df = pd.read_hdf("results-rej_per_eff-1_new.h5", "ttbar")
    print(df.columns.values)
    sig_eff = df["effs"]
    rnnip_ujets_rej = df["rnnip_ujets_rej"]
    rnnip_cjets_rej = df["rnnip_cjets_rej"]
    dips_ujets_rej = df["dips_ujets_rej"]
    dips_cjets_rej = df["dips_cjets_rej"]
    n_test = 10_000
    # it is also possible to extract it from the h5 attributes
    with h5py.File("results-rej_per_eff-1_new.h5", "r") as h5_file:
        n_test = h5_file.attrs["N_test"]
    ```

Now we can start the actual plotting part

```py
# here the plotting of the roc starts
plot_roc = roc_plot(
    n_ratio_panels=2, ylabel="background rejection", xlabel="b-jets efficiency"
)
plot_roc.add_roc(
    roc(
        sig_eff,
        rnnip_ujets_rej,
        n_test=n_test,
        rej_class="ujets",
        signal_class="bjets",
        label="RNNIP",
    ),
    reference=True,
)
plot_roc.add_roc(
    roc(
        sig_eff,
        dips_ujets_rej,
        n_test=n_test,
        rej_class="ujets",
        signal_class="bjets",
        label="DIPS",
    ),
)
plot_roc.add_roc(
    roc(
        sig_eff,
        rnnip_cjets_rej,
        n_test=n_test,
        rej_class="cjets",
        signal_class="bjets",
        label="RNNIP",
    ),
    reference=True,
)
plot_roc.add_roc(
    roc(
        sig_eff,
        dips_cjets_rej,
        n_test=n_test,
        rej_class="cjets",
        signal_class="bjets",
        label="DIPS",
    ),
)
# setting which flavour rejection ratio is drawn in which ratio panel
plot_roc.set_ratio_class(1, "ujets", label="light-flavour jets ratio")
plot_roc.set_ratio_class(2, "cjets", label="c-jets ratio")
# if you want to swap the ratios just uncomment the following 2 lines
# plot_roc.set_ratio_class(2, "ujets", label="light-flavour jets ratio")
# plot_roc.set_ratio_class(1, "cjets", label="c-jets ratio")
plot_roc.set_leg_rej_labels("ujets", "light-flavour jets rejection")
plot_roc.set_leg_rej_labels("cjets", "c-jets rejection")


plot_roc.draw()
plot_roc.savefig("roc.pdf")
```