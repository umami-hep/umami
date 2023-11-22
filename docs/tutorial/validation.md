In the final part of the tutorial series, we focus on validating the performance of the machine learning model trained in the previous section. Validation is crucial in assessing the model's generalization capabilities and ensuring its reliability. 

We'll use a specific script, `plotting_epoch_performance.py`, to calculate and plot performance metrics across training epochs.

Run the validation script using the following command:

```bash
python umami/plotting_epoch_performance.py -c examples/tutorial_jetclass/Train-config.yaml --recalculate
```
The script calculates performance metrics for each trained epoch and saves the results in a `.json` file, named with specific parameters. 
After calculation, it plots various validation metrics like accuracy, loss, and rejection per epoch.

Once the `.json` file is created, you can re-run the plotting without recalculating.

Have a look at `tagger_jetclass/plots/`. You will find various performance plots.
Please note that the performance of this classifier will not be very good, as it is trained on a very small set of features. The focus of the tutorial was to expose you to the main functionality of Umami and provide a working proof of using the JetClass dataset as input.

Please also have a look at the [evaluation](../plotting/index.md) on test samples and how to create plots that quantify the performance of trained classifiers.
