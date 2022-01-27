# LWTNN Conversion

To run the models also within [ATHENA](https://gitlab.cern.ch/atlas/athena?nav_source=navbar), it is necessary to convert the keras models to `json` files compatible with [lwtnn](https://github.com/lwtnn/lwtnn).



To convert the architecture to the `json` format used by lwtnn, the script [kerasfunc2json.py](https://github.com/lwtnn/lwtnn/blob/master/converters/kerasfunc2json.py) will be used. The usage of this script is described as
```
usage: kerasfunc2json.py [-h] arch_file hdf5_file [variables_file]

Converter from Keras saved NN to JSON

positional arguments:
  arch_file       architecture json file
  hdf5_file       Keras weights file
  variables_file  variable spec as json

optional arguments:
  -h, --help      show this help message and exit

____________________________________________________________________
Variable specification file

In additon to the standard Keras architecture and weights files, you
must provide a "variable specification" json file.

Here `scale` and `offset` account for any scaling and shifting to the
input variables in preprocessing. The "default" value is optional.

If no file is provided, a template will be generated.
```

Thus the `arch_file`, `hdf5_file` and `variables_file` need to be produced.

### Variables File

To produce the json `variables_file` a script is provided which can be executed like
```
python scripts/create_lwtnn_vardict.py -s <scale_dict.json> -v <Variables.yaml> -o lwtnn_vars.json -t <TAGGER>
```
the `--sequence_name` or `-n` option defines the required track selection described in the follwing.
Currently, 3 taggers are supported `DL1`, `DIPS` and `UMAMI`

#### Track Selection

There are different track selections [defined in Athena](https://gitlab.cern.ch/atlas/athena/blob/21.2/PhysicsAnalysis/JetTagging/FlavorTagDiscriminants/Root/DL2.cxx#L302-331) which can be used and are specified in the lwtnn json file with their definitions [here](https://gitlab.cern.ch/atlas/athena/-/blob/21.2/PhysicsAnalysis/JetTagging/FlavorTagDiscriminants/Root/DL2HighLevel.cxx#L148-156) together with the sort order of the tracks. The following options are usually used for DIPS:

* Standard: `tracks_ip3d_sd0sort`
* Loose: `tracks_dipsLoose202102_sd0sort`


### Architecture and Weight File
Splitting the model into architecture `arch_file` and weight file `hdf5_file` can be done via

```
 python scripts/conv_lwtnn_model.py -m <model.h5> -o lwtnn_model
```
This script will return two files which are in this case `architecture-lwtnn_model.json` and `weights-lwtnn_model.h5`

### Final JSON File
Finally, the three produced files can be merged via [kerasfunc2json.py](https://github.com/lwtnn/lwtnn/blob/master/converters/kerasfunc2json.py)

```
python kerasfunc2json.py architecture-lwtnn_model.json weights-lwtnn_model.h5 lwtnn_vars.json > FINAL-model.json
```

To test if the created model is properly working you can use the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper) and add the created model to a config (e.g. [EMPFlow.json](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/-/blob/master/configs/single-b-tag/EMPFlow.json)). This can exemplarily look like
```json
{
    "jet_collection": "AntiKt4EMPFlowJets_BTagging201903",
    "jet_calibration_collection": "AntiKt4EMPFlow",
    "jet_calib_file": "JES_data2017_2016_2015_Consolidated_PFlow_2018_Rel21.config",
    "cal_seq": "JetArea_Residual_EtaJES_GSC_Smear",
    "cal_area": "00-04-82",
    "do_calibration": "true",
    "run_augmenters": "false",
    "vr_cuts": "false",
    "jvt_cut": 0.5,
    "pt_cut": 20000,
    "n_tracks_to_save": 40,
    "track_sort_order": "d0_significance",
    "track_selection": {
        "pt_minimum": 1000,
        "d0_maximum": 1.0,
        "z0_maximum": 1.5,
        "si_hits_minimum": 7,
        "si_holes_maximum": 2,
        "pix_holes_maximum": 1
    },
    "dl2_configs": [
        {
            "nn_file_path": "DIPS-model.json",
            "output_remapping": {
                "DIPS_pu": "dips_pu",
                "DIPS_pc": "dips_pc",
                "DIPS_pb": "dips_pb"
            }
        }
    ],
    "variables": {
        "btag": {
            "file": "single-btag-variables.json",
            "doubles": [
                "dips_pu",
                "dips_pc",
                "dips_pb"
                ]
        },
        "track": {
            "file": "single-btag-track-variables.json"
        }
    }
}
```

To run the taggers within the dumper, we need the [r22 Branch](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/-/tree/r22) or we need to change the AnalysisBase version in the [setup.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/-/blob/master/setup.sh#L21) to `asetup AnalysisBase,22.2.12,latest`.

To run the dumper with the taggers on the grid, we need to add the path of the model file and the model itself to the job [submission file](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/-/blob/master/grid/submit.sh). Here we need to add the model path like the configs model path to the script. Also we need to give `prun` the model file as an `--extFile`, due to its size. Also you need to avoid absolute paths in the [Config File](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/-/blob/master/configs/single-b-tag/EMPFlow.json) or the grid job will fail. Add these paths to the [submission file](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/-/blob/master/grid/submit.sh).

This can look like this for example:
```
#!/usr/bin/env bash

# This script should not be sourced, we don't need anything in here to
# propigate to the surrounding environment.
if [[ $- == *i* ]] ; then
    echo "Don't source me!" >&2
    return 1
else
  # set the shell to exit if there's an error (-e), and to error if
  # there's an unset variable (-u)
    set -eu
fi


##########################
# Real things start here #
##########################

###################################################
# Part 1: variables you you _might_ need to change
###################################################
#
# Users's grid name
GRID_NAME=${RUCIO_ACCOUNT-${USER}}
#
# This job's tag (the current expression is something random)
BATCH_TAG=$(date +%F-T%H%M%S)-R${RANDOM}
# BATCH_TAG=v0

DEFAULT_CONFIG=EMPFlow_loose.json
# DEFAULT_CONFIG=TrackJets.json

DEFAULT_DIPS_LOOSE=DIPS-model-loose.json

INPUT_DATASETS=(
    mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_PHYSVAL.e6337_s3126_r12305_p4364
)

######################################################
# Part 2: variables you probably don't have to change
######################################################
#
# Build a zip of the files we're going to submit
ZIP=job.tgz
#
# This is the subdirectory we submit from
SUBMIT_DIR=submit
#
# This is where all the source files are
BASE=$(pwd)/training-dataset-dumper
#
# Configuration file stuff
DEFAULT_CONFIG_PATH=${BASE}/configs/single-b-tag/${DEFAULT_CONFIG}
DEFAULT_DIPS_LOOSE_PATH=${BASE}/configs/single-b-tag/${DEFAULT_DIPS_LOOSE}

###################################################
# Part 3: prep the submit area
###################################################
#
echo "preping submit area"
if [[ -d ${SUBMIT_DIR} ]]; then
    echo "removing old submit directory"
    rm -rf ${SUBMIT_DIR}
fi
mkdir ${SUBMIT_DIR}
CONFIG_PATH=${1-${DEFAULT_CONFIG_PATH}}
DIPS_LOOSE_PATH=${1-${DEFAULT_DIPS_LOOSE_PATH}}
echo "using config file ${CONFIG_PATH}"
echo "using Dips file ${DIPS_LOOSE_PATH}"
cp ${CONFIG_PATH} ${SUBMIT_DIR}
cp ${DIPS_LOOSE_PATH} ${SUBMIT_DIR}
# make sure we send files that the configuration depends on too
cp ${DEFAULT_CONFIG_PATH%/*}/*-variables.json ${SUBMIT_DIR}
cd ${SUBMIT_DIR}


##########################################
# Part 4: build a tarball of the job
###########################################
#
# Check to make sure you've properly set up the environemnt: if you
# haven't sourced the setup script in the build directory the grid
# submission will fail, so we check here before doing any work.
if ! type dump-single-btag &> /dev/null ; then
    echo "You haven't sourced x86*/setup.sh, job will fail!" >&2
    echo "quitting..." >&2
    exit 1
fi
#
echo "making tarball of local files: ${ZIP}" >&2
#
# The --outTarBall, --noSubmit, and --useAthenaPackages arguments are
# important. The --outDS and --exec don't matter at all here, they are
# just placeholders to keep panda from complianing.
prun --outTarBall=${ZIP} --noSubmit --useAthenaPackages\
     --exec "ls"\
     --outDS user.${GRID_NAME}.x\
     --extFile ${DEFAULT_DIPS_LOOSE}

##########################################
# Part 5: loop over datasets and submit
##########################################

# Loop over all inputs
echo "submitting for ${#INPUT_DATASETS[*]} datasets"
#
for DS in ${INPUT_DATASETS[*]}
do
   # This regex extracts the DSID from the input dataset name, so
   # that we can give the output dataset a unique name. It's not
   # pretty: ideally we'd just suffix our input dataset name with
   # another tag. But thanks to insanely long job options names we
   # use in the generation stage we're running out of space for
   # everything else.
   DSID=$(sed -r 's/[^\.]*\.([0-9]{6,8})\..*/\1/' <<< ${DS})
   #
   # Build the full output dataset name
   CONFIG_FILE=${CONFIG_PATH##*/}
   DIPS_LOOSE_FILE=${DIPS_LOOSE_PATH##*/}
   TAGS=$(cut -d . -f 6 <<< ${DS}).${CONFIG_FILE%.*}
   OUT_DS=user.${GRID_NAME}.${DSID}.btagTraining.${TAGS}.${BATCH_TAG}
   #
   # Now submit. The script we're running expects one argument per
   # input dataset, whereas %IN gives us comma separated files, so we
   # have to run it through `tr`.
   #
   echo "Submitting for ${GRID_NAME} on ${DS} -> ${OUT_DS}"
   prun --exec "dump-single-btag %IN -s -c ${CONFIG_FILE}"\
        --outDS ${OUT_DS} --inDS ${DS}\
        --useAthenaPackages --inTarBall=${ZIP}\
        --outputs output.h5\
        --noEmail > ${OUT_DS}.log 2>&1 &
   sleep 1

done
wait

```

After having the hdf5 ntuples produced, the script [`check_lwtnn-model.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/scripts/check_lwtnn_vardict.py) can be used to compare the athena evaluation with the keras evaluation. The script requiries several arguments to be correctly runned. Here is a usage example:
```
# Path to the variables yaml file
VARIABLESDICT=examples/Dips_Variables.yaml
# Path to the trained model
MODEL=trained_models/MyDipsTraining/model_epochXX.h5
# Name of the output file (optional)
ADDPATH=MyDipsTraining-diff
# Name of the tagger
TAGGER=MyDipsTraining
# Path to the prepared ntuple
HDFFILE=ftag-output.h5
# Then only one of the two following options needs to be give:
# - 1 Path to the config file used for the training
CONFIG=examples/Dips-PFlow-Training-config.yaml
# - 2 Path to the scale dictionary
SCALEDICT=MyDipsTraining_scale_dict.json


# Execute the script
python scripts/check_lwtnn-model.py -i ${HDFFILE} -v ${VARIABLESDICT} -t ${TAGGER} -m ${MODEL} -c ${CONFIG} -o ${ADDPATH}
# or
python scripts/check_lwtnn-model.py -i ${HDFFILE} -v ${VARIABLESDICT} -t ${TAGGER} -m ${MODEL} -s ${SCALEDICT} -o ${ADDPATH}
```

The output should look like, for example, to something like this:
```
Differences off 1e-6 1.34 %
Differences off 2e-6 0.12 %
Differences off 3e-6 0.03 %
Differences off 4e-6 0.02 %
Differences off 5e-6 0.01 %
Differences off 1e-5 0.0 %
```
This means that the networks scores are matching within a precision of 1e-5 for all the jets in the produced ntuple, 0.01% of the tested jets have a difference in the predicted probabilities between 5e-6 and 1e-6, and so on... Typically, we are happy when the scores match within 1e-5.