# Truth labelling

## Jet truth labels
The standard labelling is provided via the `HadronConeExclTruthLabelID` variable while an extended jet labelling is available via the `HadronConeExclExtendedTruthLabelID` variable.
For more information, consider the [FTAG TWiki about flavour labelling](https://twiki.cern.ch/twiki/bin/view/AtlasProtected/FlavourTaggingLabeling).

| HadronConeExclTruthLabelID | Category         |
| -------------------------- | ---------------- |
| 0                          | light jets       |
| 4                          | c-jets           |
| 5                          | b-jets    |
| 15                         | tau-jets         |

| HadronConeExclExtendedTruthLabelID | Category         |
| ---------------------------------- | ---------------- |
| 0                                  | light jets       |
| 4                                  | c-jets           |
| 5, 54                              | single b-jets    |
| 15                                 | tau-jets         |
| 44                                 | double c-jets    |
| 55                                 | double b-jets    |

For the `HadronConeExclTruthLabelID` labelling, the categories `4` and `44` as well as `5`, `54` and `55` are combined.

## Track truth labels

Apart from the jet labelling, also track truth origin labels are available.
The corresponding variable is `truthOriginLabel` which is defined [here](https://acode-browser1.usatlas.bnl.gov/lxr/source/athena/PhysicsAnalysis/TrackingID/InDetTrackSystematicsTools/InDetTrackSystematicsTools/InDetTrackTruthOriginDefs.h#0137).

| truthOriginLabel | Category         |
| ---------------- | ---------------- |
| 0                          | pile-up       |
| 1                          | Fake       |
| 2                          | Primary       |
| 3                          | FromB       |
| 4                          | FromBC       |
| 5                          | FromC       |
| 6                          | FromTau       |
| 7                          | OtherSecondary       |


## Vertex truth labels

`truthVertexIndex` truth vertex index of the track. 0 is reserved for the truth PV, any SVs are indexed arbitrarily with a int >0. Truth vertices within 0.1mm are merged.