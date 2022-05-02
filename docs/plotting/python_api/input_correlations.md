# Input Correlations API
Correlations between input variables can be made visible with the `input_correlations.py` script that can be found [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/plotting/input_correlations.py). It plots a linear correlation matrix and scatterplots between all variables given by a yaml variable file.

???+ example "Correlation Matrix"
    ![input_correlations](../../ci_assets/correlation_matrix.png)
    ```py linenums="71"
    §§§examples/plotting/input_correlations.py:71:111§§§
    ```

???+ example "Scatterplot Matrix"
    ![input_correlations](../../ci_assets/scatterplot_matrix.png)
    ```py linenums="114"
    §§§examples/plotting/input_correlations.py:114:182§§§
    ```
