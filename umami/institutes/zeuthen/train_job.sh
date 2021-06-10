#!/bin/zsh
#
#(otherwise the default shell would be used)
#$ -S /bin/bash
#
# This script should not be sourced, we don't need anything in here to
# propigate to the surrounding environment.
#
if [[ $- == *i* ]] ; then
    echo "Don't source me!" >&2
    return 1
else
  # set the shell to exit if there's an error (-e), and to error if
  # there's an unset variable (-u)
    set -eu
fi

own_name=$(basename $0)
tagger_name=
train_config=
model_name=
epochs=

declare -A taggers
taggers[dl1]=train_DL1.py
taggers[dips]=train_Dips.py
taggers[umami]=train_umami.py

usage() {
  echo "Usage: $own_name [-t TAGGER NAME <dl1|dips|umami>] [-c TRAIN CONFIG <path>] [-e EPOCHS <integer>]" 1>&2; exit 1;
}

while getopts ":t:c:e:m:" flag; do
    case "${flag}" in
        t) tagger_name=${OPTARG}
            [[ "$tagger_name" == "dl1" || "$tagger_name" == "dips" || "$tagger_name" == "umami" ]] || usage
            ;;
        c) train_config=${OPTARG}
            ;;
        e) epochs=${OPTARG}
            ;;
        m) model_name=${OPTARG}
            ;;
        *) usage
            ;;
    esac
done
shift $((OPTIND-1))

if [[ -z "${tagger_name}" || -z "${train_config}" ]]; then
    usage
fi

#
#############################################
#
# Options that start with #$s will be applied.
# For more details, man qsub.
# If running this script as a stand-alone,
# submit the script using qsub:
# qsub train_job.sh -t dl1 -c training-config.yml -e 200
##############################################
#
#(request gpu, this specific type is compatible with the latest umami singularity container)
#$ -l gpu_type=nvidia_geforce_rtx_3090
#
#(job's maximum time)
#$-l h_rt=1:00:00
#
#(job's maximum resident memory usage)
#$ -l h_rss=15G
#
#(job's maximum scratch space usage in $TMPDIR)
#$ -l tmpdir_size=25G
#
#(stderr and stdout are merged together to stdout)
#$ -j y
#
#(execute the job from the current directory and not relative to your home directory)
#$ -cwd
#

epochs_option=""
if [[ ! -z "${epochs}" ]]; then
    epochs_option="-e ${epochs}"
fi

# singularity should cache in $TMPDIR
SINGULARITY_CACHEDIR=$TMPDIR/.singularity
train_config_path=$SGE_O_WORKDIR/$train_config

cd $TMPDIR
singularity exec --nv docker://btagging/umami:latest-gpu python /umami/umami/${taggers[$tagger_name]} -c ${train_config_path} ${epochs_option}

# copy results to working directory on client
if [ -z "${model_name}" ]; then
    # copy most recently created directory, assuming it's the model
    model_name=$(ls -td -- */ | head -n 1)
fi
cp -r $model_name $SGE_O_WORKDIR/
