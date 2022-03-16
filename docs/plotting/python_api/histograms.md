# Histogram plotting API

## Common plots

In the following a small example how to plot histograms with the umami python api.

To set up the inputs for the plots, have a look [here](./index.md).

The following examples use the dummy data which is described [here](./dummy_data.md)


???+ example "Discriminant scores"
    ![discriminant](../../ci_assets/histogram_discriminant.png)
    ```py linenums="1"
    §§§examples/plotting/plot_discriminant_scores.py§§§
    ```


???+ example "Raw tagger output (flavour probabilities)"
    ![b-jets probability](../../ci_assets/histogram_bjets_probability.png)
    ```py linenums="1"
    §§§examples/plotting/plot_flavour_probabilities.py§§§
    ```

## Example for basic untypical histogram

In most cases you probably want to plot histograms with with the different flavours
like in the examples above.
However, the python plotting API allows to plot any kind of data. As an example, you
could e.g. produce a `MC` vs `data` plot with the following example code:


???+ example "More general example"
    ![non-ftag example](../../ci_assets/histogram_basic_example.png)
    ```py linenums="1"
    §§§examples/plotting/plot_basic_histogram.py§§§
    ```
