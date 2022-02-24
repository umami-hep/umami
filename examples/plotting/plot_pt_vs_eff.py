"""Produce pT vs efficiency plot from tagger output and labels."""
import h5py
import numpy as np
import pandas as pd

from umami.configuration import logger
from umami.plotting import var_vs_eff, var_vs_eff_plot

# this is just an example to read in your h5 file
# if you have tagger predictions you can plug them in directly in the `disc_fct` as well

# using a random ttbar file
ttbar_file = (
    "user.alfroch.410470.btagTraining.e6337_s3681_r13144_p4931.EMPFlowAll."
    "2022-02-07-T174158_output.h5/user.alfroch.28040424._001207.output.h5"
)

logger.info("load file")
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
                "pt",
            ]
        )[:300000]
    )
logger.info("caclulate tagger discriminants")


# define a small function to calculate discriminant
def disc_fct(a: np.ndarray) -> np.ndarray:
    """Tagger discriminant

    Parameters
    ----------
    a : numpy.ndarray
        array with with shape (, 3)

    Returns
    -------
    np.ndarray
        Array with the discriminant values inside.
    """
    # you can adapt this for your needs
    return np.log(a[2] / (0.018 * a[1] + 0.92 * a[0]))


# you can also use a lambda function
# discs_rnnip = np.apply_along_axis(
#     lambda a: np.log(a[2] / (0.018 * a[1] + 0.92 * a[0])),
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

# you can also use a results file directly, you can comment everything above and
# uncomment below
# ttbar_file = "<resultsfile.h5"
# df = pd.read_hdf(ttbar_file, key="ttbar")

# discs_rnnip = df["disc_rnnip"]
# discs_dips = df["disc_dips"]
# is_light = df["labels"] == 0
# is_c = df["labels"] == 1
# is_b = df["labels"] == 2

# Getting jet pt in GeV
pt = df["pt"].values / 1e3
# defining target efficiency
sig_eff = np.linspace(0.49, 1, 20)
# defining boolean arrays to select the different flavour classes
is_light = df["HadronConeExclTruthLabelID"] == 0
is_c = df["HadronConeExclTruthLabelID"] == 4
is_b = df["HadronConeExclTruthLabelID"] == 5

# here the plotting starts

# define the curves
rnnip_light = var_vs_eff(
    x_var_sig=pt[is_b],
    disc_sig=discs_rnnip[is_b],
    x_var_bkg=pt[is_light],
    disc_bkg=discs_rnnip[is_light],
    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
    wp=0.7,
    disc_cut=None,
    fixed_eff_bin=False,
    label="RNNIP",
)
dips_light = var_vs_eff(
    x_var_sig=pt[is_b],
    disc_sig=discs_dips[is_b],
    x_var_bkg=pt[is_light],
    disc_bkg=discs_dips[is_light],
    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
    wp=0.7,
    disc_cut=None,
    fixed_eff_bin=False,
    label="DIPS",
)


logger.info("Plotting bkg rejection for inclusive efficiency as a function of pt.")
# You can choose between different modes: "sig_eff", "bkg_eff", "sig_rej", "bkg_rej"
plot_bkg_rej = var_vs_eff_plot(
    mode="bkg_rej",
    ylabel="Light-flavour jets rejection",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
)
plot_bkg_rej.add(rnnip_light, reference=True)
plot_bkg_rej.add(dips_light)

plot_bkg_rej.draw()
plot_bkg_rej.savefig("pt_light_rej.pdf")
# clearing the plot
plot_bkg_rej.clear()

plot_sig_eff = var_vs_eff_plot(
    mode="sig_eff",
    ylabel="b-jets efficiency",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
)
plot_sig_eff.add(rnnip_light, reference=True)
plot_sig_eff.add(dips_light)

plot_sig_eff.atlas_second_tag += "\nInclusive $\\epsilon_b=70%%$"

# If you want to inverse the discriminant cut you can enable it via
# plot_sig_eff.set_inverse_cut()
plot_sig_eff.draw()
# Drawing a hline indicating inclusive efficiency
plot_sig_eff.draw_hline(0.7)
plot_sig_eff.savefig("pt_b_eff.pdf")
# clearing the plot
plot_sig_eff.clear()
