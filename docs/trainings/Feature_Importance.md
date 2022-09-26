# Explaining the importance of features with SHAPley

[SHAPley](https://github.com/slundberg/shap) is a framework that helps you understand how your training of a machine learning model is affected by the input variables, or in other words from which variables your model possibly learns the most. SHAPley is for now only usable when evaluating a DL1* version. You can run that by executing the command

```bash
evaluate_model.py -c examples/training/DL1r-PFlow-Training-config.yaml -e 230 -s shapley
```

which will output a beeswarm plot into `modelname/plots/`. Each dot in this plot is for one whole set of features (or one jet). They are stacked vertically once there is no space horizontally anymore to indicate density. The colour map tells you what the actual value was that entered the model. The SHAP value is basically calculated by removing features, letting the model make a prediction and then observe what would happen if you introduce features again to your prediction. If you do this over all possible combinations you get estimates of a features impact to your model. This is what the x-axis (SHAP value) tells you: the on average(!) contribution of a variable to an output node you are interested in (default is the output node for $b$-jets). In practice, large magnitudes (which is also what these plots are ordered by default in umami) are great, as they give the model a better possibility to discriminate. Features with large negative SHAP values therefore will help the model to better reject, whereas features with large positive SHAP values helps the model to learn that these are most probably jets from the category of interest. If you want to know more about SHAPley values, here is a [talk](https://indico.cern.ch/event/1071129/#4-shapely-for-nn-input-ranking) from one of our FTAG algorithm meeting.

You have some options to play with in the `evaluation_settings` section in the [DL1r-PFlow-Training-config.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/training/DL1r-PFlow-Training-config.yaml) shown here:

```yaml
§§§examples/training/DL1r-PFlow-Training-config.yaml:157:178§§§
```
