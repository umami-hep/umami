???+ question "I ran the plotting_epoch_performance.py script and get the a KeyError for one of my validation samples."



    First of all, check that the your validation samples are correctly defined in the train config. After that, check if you ran with on-the-fly validation off (In `validation_settings`, the `n_jets` are 0). If that's the case, you need to re-run the calculation of the validation metrics via 

    ```bash
    plotting_epoch_performance.py -c <path to train config file> --recalculate
    ```

    This will re-validate the defined validation samples and will output a new validation `.json`.

???+ question "I added a new validation sample, but I get a KeyError for that when running the plotting_epoch_performance.py script."



    To add a new validation sample to your validation `.json` (where the values are stored which are plotted), you need to re-run the calculation
    of these values. This can be done by running the following command:

    ```bash
    plotting_epoch_performance.py -c <path to train config file> --recalculate
    ```

    **Note** This will overwrite also an already existing `.json`. So please keep all validation samples you want to use in the train config when re-validating.