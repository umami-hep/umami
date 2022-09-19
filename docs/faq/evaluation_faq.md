???+ question "I have a sample and want to check the performance of the taggers for which the values are already stored in the files"



    If you want to just check the performance of one or more taggers which are already present in the `.h5` files, you can
    follow the steps explained [here](https://umami-docs.web.cern.ch/trainings/evaluate/#evaluate-only-the-taggers-inside-the-h5-files-without-a-freshly-trained-model).




???+ question "I have a new sample and I want to evaluate the performance of my trained tagger on that"



    To do that, you need to follow these steps:

    - Add a new entry to `test_files` in your train config file with the path to the sample and the cuts you want to apply (Wildcards are supported).
    - Re-run the `evaluate_model.py` script (see [here](https://umami-docs.web.cern.ch/trainings/evaluate/#running-the-evaluation) for explanation how to do that).
    - In your results files is now a new entry with the name of the entry you set in `test_files`.
    - You should be able to plot the results now using the `plotting_umami.py` scripts (see [here](https://umami-docs.web.cern.ch/plotting/) for explanation how to do that) or your personal [`PUMA`](https://github.com/umami-hep/puma/) scripts.