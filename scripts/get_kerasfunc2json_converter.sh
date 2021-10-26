#!/bin/bash

# get lwtnn
git clone git@github.com:lwtnn/lwtnn.git

# add lwtnn/converters to pythonpath
export PYTHONPATH=${PYTHONPATH}:lwtnn/converters/

echo "Obtained ltwnn from github, you can now run python lwtnn/converters/kerasfunc2json.py"