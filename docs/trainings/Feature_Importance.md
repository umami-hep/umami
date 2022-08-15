# Explaining the importance of features with SHAPley

[SHAPley](https://github.com/slundberg/shap) is a framework that helps you understand how your training of a machine learning model is affected by the input variables, or in other words from which variables your model possibly learns the most. You just need to add a `--shapley` flag to `evaluate_model.py --tagger dl1` as e.g.

```bash
python umami/evaluate_model.py -c examples/training/DL1r-PFlow-Training-config.yaml -e 230 --tagger dl1 --shapley
```

and it will output a beeswarm plot into `modelname/plots/`. Each dot in this plot is for one whole set of features (or one jet). They are stacked vertically once there is no space horizontally anymore to indicate density. The colormap tells you what the actual value was that entered the model. The Shap value is basically calculated by removing features, letting the model make a prediction and then observe what would happen if you introduce features again to your prediction. If you do this over all possible combinations you get estimates of a features impact to your model. This is what the x-axis (SHAP value) tells you: the on average(!) contribution of a variable to an output node you are interested in (default is the output node for b-jets). In practice, large magnitudes (which is also what these plots are ordered by default in umami) are great, as they give the model a better possibility to discriminate. Features with large negative shap values therefore will help the model to better reject, whereas features with large positive shap values helps the model to learn that these are most probably jets from the category of interest. If you want to know more about shapley values, here is a [talk](https://indico.cern.ch/event/1071129/#4-shapely-for-nn-input-ranking) from our alorithms meeting.

You have some options to play with in the `Eval_parameters_validation` section in the [DL1r-PFlow-Training-config.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/training/DL1r-PFlow-Training-config.yaml)

```yaml
Eval_parameters_validation:

    (...other stuff...)

    shapley:
        # Over how many full sets of features it should calculate over.
        # Corresponds to the dots in the beeswarm plot.
        # 200 takes like 10-15 min for DL1r on a 32 core-cpu
        feature_sets: 200

        # defines which of the model outputs (flavor) you want to explain
        # [tau,b,c,u] := [3, 2, 1, 0]
        model_output: 2

        # You can also choose if you want to plot the magnitude of feature
        # importance for all output nodes (flavors) in another plot. This
        # will give you a bar plot of the mean SHAP value magnitudes.
        bool_all_flavor_plot: False

        # as this takes much longer you can average the feature_sets to a
        # smaller set, 50 is a good choice for DL1r
        averaged_sets: 50

        # [11,11] works well for dl1r
        plot_size: [11, 11]
```
