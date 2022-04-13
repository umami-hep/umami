# Variable vs efficiency plots
In the following a small example how to plot the efficiency vs a specific variable
with the umami python api. In this case, we use `pt` as this variable.

To set up the inputs for the plots, have a look [here](./index.md).

Then we can start the actual plotting part


???+ example "pT vs efficiency and pT vs rejection plot"

    ![pt_vs_eff](../../ci_assets/pt_light_rej.png)

    ![pt_vs_eff](../../ci_assets/pt_b_eff.png)

    ```py linenums="1"
    §§§examples/plotting/plot_pt_vs_eff.py§§§
    ```