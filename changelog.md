# Changelog

### Latest

- Added writing validation files [!659](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/659)
- Adding Appache 2.0 license [!656](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/656)
- Fixing issue in the try except blocks of the preprocessing plots [!655](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/655)
- Setting default value for concat_jet_tracks [!654](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/654)
- Adding support for non-top level and special named jet- and track collections [!653](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/653)
- Various improvements to train file writing (group-based structure) [!648](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/648)
- Removing file dependency for generator unit tests [!652](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/652)

### [v0.14](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/tags/0.14) (14.10.2022)

- Moving Feature Importance in the evaluation section of the training [!647](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/647)
- Fixing issue with SHAPley plot naming [!645](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/645)
- Adding proper hybrid validation sample creation [!646](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/646)
- Fixing SHAPley calculation and add it to DL1r integration test [!643](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/643)
- Adding `yaml` to `requirements.txt` [!644](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/644)
- Adding classes_to_evaluate to rejection per fraction calculation [!642](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/642)
- Adding function to write model predictions to h5 files [!637](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/637)
- Fixing ufunc issue in scale/shift application [!641](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/641)
- Adding FAQ and small docs update [!636](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/636)

### [v0.13](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/tags/0.13) (15.09.2022)

- Fix error calculation in ROC plots [!639](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/639)
- Remove global `dropout` parameter from DIPS config. Dropout in DIPS is now defined for each layer with `dropout_rate` and `dropout_rate_phi` [!638](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/638)
- Re-adding `#!/usr/bin/env python` to executable scripts [!635](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/635)
- Removing global `dropout` parameter from DL1* models. Dropout has to be specified per layer now via the list `dropout_rate` [!633](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/633)
- Bot-comment about changed placeholders will now be posted as unresolved thread [!634](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/634)
- Adding function to flatten arbitrary nested lists [!632](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/632)
- Adding randomise option to input_h5 block in preprocessing config [!631](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/631)
- Input variable plots: Adding support for custom x-labels [!626](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/626)
- Input variable plots: Adding support for dataset-specific class labels [!623](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/623)
- Adding possibility to evaluate classes the freshly trained tagger is not trained on [!625](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/625)
- Remove preprocessing config from loading functions [!622](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/622)
- Training metrics plots: now using `puma.Line2DPlot` objects here, which modifies the default colours [!629](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/629)
- Fixing plotting issue in fraction contour + plot_scores [!624](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/624)
- Switch to puma v0.1.8 [!630](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/630)
- Adding support for dataset-specific class labels in input var plots [!623](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/623)
- Apply naming scheme for WP and nEpochs [!621](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/621)
- Adding correct naming scheme for train config sections [!617](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/617)

### [v0.12](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/tags/0.12) (23.08.2022)

- Small resampling fix [!620](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/620)
- Fixing multiple issues with the fraction contour plots [!619](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/619)
- Adding automatic creation of samples dict for the preprocessing config [!610](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/610)
- Rewriting of preprocessing config reader [!606](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/606)
- Adding truth label to results file + Fix flavour retrieval in plotting [!618](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/618)
- Merging load validation data functions [!615](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/615)
- Update training documentation [!613](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/613)
- Cleanup of preprocessing config [!609](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/609)
- Update puma version to v0.1.7 [!614](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/614)

### [v0.11](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/tags/0.11) (10.08.2022)

- Removing var_dict from train config [!611](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/611)
- Switching track variable precision to float32 [!608](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/608)
- Merge apply_scaling and write step in preprocessing [!605](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/605)
- Adding string join support for yaml [!607](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/607)
- Adding configuration base class and doc improvements for pdf sampling [!604](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/604)
- Merging evaluate_model script funtions + adapt pt_vs plots to be var_vs plots [!599](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/599)
- Unify scaling/shifting application for preprocessing/validation [!597](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/597)
- Adding script to process test samples in an easy way  [!595](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/595)
- Adding x_axis_granularity argument + Fixing evaluation_file plotting issue [!596](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/596)
- Restructure and update preprocessing documentation [!598](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/598)
- Bot posts message in MR in case files used as placeholders were changed [!594](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/594)
- Pointing truth label docs directly to FTAG docs  [!593](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/593)
- Compare class id, class operators and variables of each class definition instead of only comparing the class id to avoid the same class definition. [!575](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/575)
- Removing #!/usr/bin/env python from scripts  [!591](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/591)
- Adding metadata information to training file [!592](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/592)
- Adding some missing unit tests [!587](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/587)
- Plots per default with non-transparent background [!590](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/590)
- Fixing pylint for unit tests [!588](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/588)
- Adding support for hits [!583](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/583)
- Fixing track masking for the input variable plots [!585](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/585)
- Reducing artifact size for the preprocessing integration tests [!586](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/586)
- Removing casefold in tagger name retrieval [!584](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/584)
- Fixing all pylint `logging-fstring-interpolation` issues [!582](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/582)
- Adding consistent n_jets naming [!570](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/570)

### [v0.10](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/tags/0.10) (06.07.2022)

- Adding track truth label to the Preprocessing. [!559](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/559)
- Fixing CI syntax of `cobertura` [!577](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/577)
- Fixing image issue in pylint [!574](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/574)
- Fixing memory leak in Callback functions + New TF version 2.9.1 [!573](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/573)
- Add option `sampling_fraction` in preprocessing config to use a different number of jets for each class. Defined as fraction of events compared to target class, add option to define operator in global config [!561](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/561)
- Switch to latest puma version (v0.1.3) [!572](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/572)
- Splitting CADS and DIPS Attention [!569](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/569)
- Fixing docker image builds [!571](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/571)
- Fixing uncertainty calculation for the ROC curves [!566](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/566)

### [v0.9](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/tags/0.9) (21.06.2022)

- Fixing Callback error when LRR is not used [!567](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/567)
- Fixing stacking issue for the jet variables in the PDFSampling [!565](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/565)
- Fixing problem with 4 classes integration test [!564](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/564)
- Rework saliency plots to use puma [!556](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/556)
- Fixing generation of class ids for only one class [!563](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/563)
- Removing hardcoded tmp directories in the integration tests [!562](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/562)
- Fixing x range in metrics plots + correct tagger name in results files [!560](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/560)
- Fixing issue with the PDFSampling shuffling + Fixing small issue with the loaders [!558](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/558)
- Fixing ylabel issue in ROC plots [!555](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/555)
- Adding verbose option to executable scripts [!557](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/557)
- Moving Plotting Files in one folder [!554](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/554)
- Adding classes to global config (light-flavour jets split by quark flavour/gluons, leptonic b-hadron decays) to define extended tagger output [!553](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/553)
- Fixing issues with trained_taggers and taggers_from_file in plotting_epoch_performance.py [!549](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/549)
- Adding plotting API to Contour plots + Updating plotting_umami docs [!537](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/537)
- Adding unit test for prepare_model and minor bug fixes [!546](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/546)
- Adding unit tests for tf generators[!542](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/542)
- Fix epoch bug in continue_training[!543](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/543)
- Updating tensorflow to version `2.9.0` and pytorch to `1.11.0-cuda11.3-cudnn8-runtime` [!547](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/547)
- Removing plotting API code and switch to puma [!540](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/540) [!548](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/548)
- Fix epoch bug in continue_training[!543](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/543)
- Remove IPxD from default configs [!544](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/544)

### [v0.8](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/tags/0.8) (16.05.2022)

- Fix integration test artifacts [!538](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/538)
- Moving the line-block replacement script to a separate repo [!539](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/539)
- Apply Plotting API to preprocessing plots[!534](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/534)
- Adding fix for batch size in validation/evaluation [!535](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/535)
- Adding Plotting API to PlottingFunctions in the eval tools [!532](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/532)
- Fix for the "exclude" funtionality [!528](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/528)
- Adding metrics to Callback functions + Fixing model summary issue [!526](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/526)
- Improved compression settings during scaling and writing [!527](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/527)
- Add documentation and integration tests for importance sampling without replacement method [!502](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/502)
- (Plotting API) Update training plots to plotting API [!515](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/515)
- Fix validation values json in continue_training [!516](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/516/)
- Fixing bunch of invalid-name pylint errors [!522](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/522)
- Adding error message if file in placeholder does not exist [!519](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/519)
- Update the LWTNN scripts [!512](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/512)
- Adding pydash to requirements [!517](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/517)
- (Plotting API) Change default value of atlas_second_tag [!514](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/514)
- Small refinements in input var plots [!505](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/505)
- Adding ylabel_ratio_1 and ylabel_ratio_2 to plot_base [!504](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/504)
- Adding prepare_docs stage to CI [!503](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/503)
- Extend flexibility in input var plotting functions [!501](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/501)
- Adding continue_training option [!500](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/500)
- change default fc for evaluation of Dips and Cads in training configs [!499](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/499)
- Use plotting python API in input var plots (track variables) [!498](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/498)
- Remove redundant loading loop [!496](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/496)
- Use plotting python API in input var plots (track variables) [!488](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/488)
- Fixing nFiles for tfrecords training [!495](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/495/)
- (Plotting API) Adding support for removing "ATLAS" branding on plots [!494](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/494)
- (Plotting API) Adding option to specify number of bins (instead of bin edges) in histogram plots [!491](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/491)
- (Plotting API) Adding support for ATLAS tag offset + Small fix for ratio uncertainty in histogram plots [!490](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/490)
- Adding support for multiple signal classes [!414](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/414)

### [v0.7](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/tags/0.7) (18.03.2022)

- Adding Script for input variables correlation plots to examples folder [!474](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/474)
- Adding integration tests for plotting examples scripts + added plots to documentation [!480](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/480)
- Adding slim umami image (mainly for plotting) [!473](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/473) [!482](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/482)
- Update python packaging, fixing CI gitlab labels and moving `classification_tools` into `helper_tools` [!481](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/481)
- Added histogram plots to the new plotting python API [!449](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/449)
- Implemented placeholder for code snippets in markdown files [!476](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/476)
- Fixing branch unit test (problem with changing style of matplotlib globally) [!478](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/478)
- Streamline h5 ntuples and samples overview with that of ftag-docs [!479](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/479)
- Adding dummy data generation of multi-class classification output [!475](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/475)
- Move to `matplotlib.figure` API and `atlasify` for plotting python API [!464](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/464)
- Adding `--prepare` option to `train.py` and fix an issue with the `model_file` not copied into the metadata folder [!472](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/472)
- Move to `matplotlib.figure` API and `atlasify` for plotting python API [!464](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/464)
- Fixing issue [#157](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/issues/157) with the `ylabel` of the input variable plots [!466](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/466).
- Adding custom labels for the `taggers_from_files` option in the validation metrics plots.
- Adding custom labels for the `taggers_from_files` option in the validation metrics plots [!469](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/469).
- Fixing doubled integration test and removing old namings [!455](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/455)
- Adding new instructions for VS Code usage [!467](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/467)
- Fixing `fixed_eff_bin` for pT dependence plots and adding new feature to set the y limit of the ratio plots for the ROC plots [!465](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/465)
- Adding a check for `replaceLineInFile` if leading spaces stay same, if not a warning is raised [!451](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/451)
- Allowing that no cuts are provided for samples in the preprocessing step [!451](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/451)
- Updating jet training variable from `SV1_significance3d` to `SV1_correctSignificance3d` for r22 [!451](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/451)
- Restructuring gitlab CI file structure and adding MR/issue templates [!463](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/463)
- Removing spectator variables from variable configs and fixing `exclude` option in training [!461](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/461)
- Adding `atlasify` to requirements [!458](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/458)
- Supprting binariser for 2 class labels to have still one hot encoding [!409](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/409)
- Variable plots for preprocessing stages added [!440](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/440)
- Update TFRecord reader/writer + Adding support for CADS and Umami Cond Att [!444](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/444)
- Restructuring documentation  [!448](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/448)
- New Python API for plotting of variable vs efficenciy/rejection  [!434](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/434)
- New combine flavour method for PDF sampling (with shuffling) [!442](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/442)
- Add TFRecords support for CADS [!436](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/436)
- Added Umami attention [!298](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/298)
- renamed `nominator` to `numerator` [!447](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/447)
- Fix of calculation of scaling factor [!441](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/441)

### [v0.6](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/tags/0.6) (16.02.2022)

- CI improvements
- latest samples added to documentation
- packages were upgraded
- new Python API added for plotting of ROC curves
- Added normalisation option to input plotting
- logging level for all tests are set by default to debug
- Added optional results filename extension
- Added docs for pdf method and parallelise pdf method
- Possibility to modify names of track variables in config files
- Added new sphinx documentation
- Black was added in CI
- fraction contour plots were added
- bb-jets category colour was changed
- Copying now config files during pre-processing
- several doc string updates
- docs update for taggers (merged them)
- save divide added
- flexible validation sample definition in config added
- fixed all doc strings and enforce now darglint in CI

### [v0.5](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/tags/0.5) (26.01.2022)

- Adding Multiple Tracks datasets in preprocessing stage in [!285](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/285)

### [v0.4](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/tags/0.4) (25.01.2022)

- Updating `Tensorflow` version from `2.6.0` to `2.7.0`
- Upgrading Python from version `3.6.9` to `3.8.10`
- Adding new base and baseplus images
- Introducing linting to the CI pipelines
- Changing to Pylint as main linting package
- Adding doc-string checks (not enforced)
- Adding support for GNN preprocessing
- Restructuring of the training config files
- Explanation how to set up Visual Studio Code to develop Umami
- Automatic documentation via `sphinx-docs` is added
- Reordering of the preprocessing config file structure (NO BACKWARD COMPATABILITY)
- Adding CI pipeline updates
- Restructuring of functions (where they are saved)
- Adding multiple updates for the taggers (mostly minor adds, no big change in performance is expected)

### [v0.3](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/tags/0.3) (01.12.2021)

- new preprocessing chain included
- adding PDF sampling, weighting
