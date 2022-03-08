# Changelog



### Latest

- Removing spectator variables from variable configs and fixing `exclude` option in training [!461](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/461)
- Adding `atlasify` to requirements [!458](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/458)
- Updating jet training variable from `SV1_significance3d` to `SV1_correctSignificance3d` for r22 [!451](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/451)
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



### [v0.6](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/tags/0.6)

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


### [v0.5](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/tags/0.5)

- Adding Multiple Tracks datasets in preprocessing stage in [!285](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/285)

### [v0.4](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/tags/0.4)

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


### [v0.3](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/tags/0.3)

- new preprocessing chain included
- adding PDF sampling, weighting