#!/usr/bin/bash
# activate python env via conda pack
# Define the working directory
RTSLM_STORAGE_DIR=$(pwd)/storage #TODO: revise to your own rtslm dir for storage (the repo downloaded from huggingface)
RTSLM_WORK_DIR=$(pwd) # TODO: revise to your own rtslm working dir (the repo clone from github)
CONDA_ACTIVATION_SOURCE=$(pwd)/taste_env/bin/activate # TODO: revise to your own conda env fpath

source $CONDA_ACTIVATION_SOURCE
export RTSLM_WORK_DIR=$RTSLM_WORK_DIR
export RTSLM_STORAGE_DIR=$RTSLM_STORAGE_DIR

echo "RTSLM_STORAGE_DIR=$RTSLM_STORAGE_DIR"
echo "RTSLM_WORK_DIR=$RTSLM_WORK_DIR"
DIRS="$RTSLM_WORK_DIR/CosyVoice/third_party/Matcha-TTS $RTSLM_WORK_DIR/CosyVoice $RTSLM_WORK_DIR/SpokenLM" # separated by space

# Check if the directory is already in PYTHONPATH
for DIR in $DIRS; do
    if [[ ":$PYTHONPATH:" != *":$DIR:"* ]]; then
        # If not, append it to PYTHONPATH
        export PYTHONPATH="${PYTHONPATH:+"$PYTHONPATH:"}$DIR"
        echo "$DIR has been added to PYTHONPATH."
    else
        echo "$DIR is already in PYTHONPATH."
    fi
done
