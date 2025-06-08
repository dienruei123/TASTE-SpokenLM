# TASTE

This README will cover how to train and use the TASTE speech tokenizer (https://arxiv.org/abs/2504.07053)

## Setting up the environment

We use conda-pack to wrap our pre-set environment and required pre-trained models and have uploaded it to huggingface. We will teach you how to download and prepare the env and models for later usage. 

### download the required files

```bash
# download cuda environment
python storage/download_env.py

# download pre-trained models
python storage/download_pretrained.py

# download data
python storage/download_data.py
```

### prerequisite

You have to install

1. miniconda (https://www.anaconda.com/docs/getting-started/miniconda/main)
2. conda pack (https://conda.github.io/conda-pack/)

### prepare the env (with conda pack)

Referencing conda-pack (https://conda.github.io/conda-pack/), you can set up the env by:

```bash
# 0. cd into the rtslm work dir
cd /path/to/your/rtslm
# 1. make dir for your own taste env
mkdir -p $(pwd)/taste_env 
# 2. extract the downloaded env file to your taste_env dir
tar -xzvf ./storage/env/taste_env.tar.gz -C $(pwd)/taste_env
# 3. now you can manually activate the env by:
source $(pwd)/taste_env/bin/activate
```
Then, please follow the instructions below to set up some required env variables:
```bash
# 1. checkout rtslm/path.sh and modify some env variables (RTSLM_STORAGE_DIR, RTSLM_STORAGE_DIR, CONDA_ACTIVATION_SOURCE)
# 2. source the path.sh
source ./path.sh # note that you should have activated the conda env after properly setting up the env
# 4. after that, install some additional packages:
pip install transformers
pip install $RTSLM_WORK_DIR/CosyVoice/third_party/FunASR
pip install einx
```

## Training TASTE Speech Tokenizer

We gave a simple demo of training TASTE on our prepared dataset for example.

```bash
# please make sure you have sourced path.sh and set the env variables properly.
cd $RTSLM_WORK_DIR/CosyVoice/examples/emilia/taste
# before actually running the scripts, please visit the configuration files under
# `rtslm/CosyVoice/examples/emilia/taste/conf`, and modify the config for your own need. 
# e.g., the `whisper_tokenizer_fpath` in the taste_no_vq.yaml and taste.yaml requires resetting. 
# The training related hyperparameters should also be modified based on your computational resources. 
bash run_train_taste.sh
```
You can download and find the larger dataset we use to train our model at: TODO

## Evaluating TASTE Speech Tokenizer

Under `$RTSLM_WORK_DIR/CosyVoice/examples/emilia/taste`, we have put three evaluation examples for:

1. Evaluating the text-only baseline:
    ```bash
    cd $RTSLM_WORK_DIR/CosyVoice/examples/emilia/taste
    # NOTE that you should modify the `EXP_NAME` under each eval scripts for your own need. 
    bash eval_text.sh
    ```

2. Evaluating TASTE:
    ```bash
    cd $RTSLM_WORK_DIR/CosyVoice/examples/emilia/taste
    # NOTE that you should modify the `EXP_NAME` under each eval scripts for your own need. 
    bash eval_taste.sh
    ```

3. Evaluating the S3 token topline:
    ```bash
    cd $RTSLM_WORK_DIR/CosyVoice/examples/emilia/taste
    # NOTE that the `EXP_NAME` for eval_s3 is dummy. Just provide a valid one for loading the llm.pt. The llm will be skipped during the inference.    
    bash eval_s3.sh
    ```

## Convert your own TASTE to TASTE-SpokenLM HF Format

The default config from both sides is set to be aligned with each other. 
After setting up the [TASTE-SpokenLM repo](https://github.com/mtkresearch/TASTE-SpokenLM), you can convert the speech tokenizer and speech decoder easily by running the python scripts: 

```bash
# under rtslm/CosyVoice/examples/emilia/taste:
python convert_to_hf_compatible.py # need to modify several arguments in the file. 
```

## Download checkpoints of TASTE

```bash
python storage/download_checkpoints.py
```

---
## Acknowledge

1. We develop TASTE based on [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) and [Whisper](https://github.com/openai/whisper).
4. We borrow a lot of code from [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS).
2. We borrow a lot of code from [FunASR](https://github.com/modelscope/FunASR).

## Disclaimer
The content provided above is for academic purposes only and is intended to demonstrate technical capabilities. Some examples are sourced from the internet. If any content infringes on your rights, please contact us to request its removal.
