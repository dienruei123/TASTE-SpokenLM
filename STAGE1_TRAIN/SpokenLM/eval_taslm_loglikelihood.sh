#!/usr/bin/bash
# NOTE: please run this script under xxxx/emilia/taste
source ~/.bashrc
source $COSYENV

WORK_DIR=/proj/mtklmadm/dev/mtk53678
cd $WORK_DIR/rtslm/SpokenLM
GPU=2

# baseline
# taslm_pretrained_dir=$WORK_DIR/rtslm/SpokenLM/taslm/exp/0211_baseline_1B_s3-pad
# speech_tokenizer_pretrained_dir=/proj/mtklmadm/dev/mtk53678/rtslm_storage/pretrained_models/CosyVoice-300M # s3

# taslm
taslm_pretrained_dir=$WORK_DIR/rtslm/SpokenLM/taslm/exp/0213_taslm_1B_eos-rvq_word-delay_masked
speech_tokenizer_pretrained_dir=$WORK_DIR/rtslm/CosyVoice/examples/emilia/taste/exp/llm/torch_ddp/stage2/0207_stg2_eos_rvq-d256-l4-k512_sum_smaller-lr # does not need to specify if it is taste tokenizer

CUDA_VISIBLE_DEVICES=$GPU python scripts/eval_taslm_loglikelihood.py \
    --pretrained_dir $taslm_pretrained_dir \
    --speech_tokenizer_pretrained_dir $speech_tokenizer_pretrained_dir
