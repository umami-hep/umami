# ROC curve plotting API
In the following a small example how to plot a roc curve with the umami python api.

To set up the inputs for the plots, have a look [here](../index.md).

Then we can start the actual plotting part

```py
from umami.plotting import roc, roc_plot
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