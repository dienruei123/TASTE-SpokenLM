#!/bin/bash
set -e

# please unsure you are under rtslm/CosyVoice/examples/emilia/taste
source ../../../../path.sh


PRETRAINED_DIR=$RTSLM_STORAGE_DIR/pretrained_models/CosyVoice-300M

SEED=1986
EXP_ROOT=./exp/llm/torch_ddp
EXP_NAME=taste_pretrained
SAMPLING=10
EXP_DIR=$EXP_ROOT/$EXP_NAME

CKPT_NAME=checkpoint_best
SPLIT=emilia-examples
OUTPUT_ROOT=./results
OUTPUT_DIR=$OUTPUT_ROOT/$SPLIT/$EXP_NAME/$CKPT_NAME

test_list_fpath=./eval/test-audio_${SPLIT}.tsv
config_fpath=$EXP_DIR/config.yaml
llm_fpath=$EXP_DIR/$CKPT_NAME.pt
flow_fpath=$PRETRAINED_DIR/flow.pt
hift_fpath=$PRETRAINED_DIR/hift.pt
python $RTSLM_WORK_DIR/CosyVoice/cosyvoice/bin/eval_audio.py \
    --config $config_fpath \
    --audio_text_list $test_list_fpath \
    --llm_fpath $llm_fpath \
    --flow_fpath $flow_fpath \
    --hift_fpath $hift_fpath \
    --model_dir $PRETRAINED_DIR \
    --gpu 1 \
    --seed $SEED \
    --output_dir $OUTPUT_DIR \
    --whisper_tokenizer_dir $RTSLM_STORAGE_DIR/pretrained_models/distil-whisper-large-v3 \
    --copy_src \
    --sampling $SAMPLING \
    --normalize_and_resample_source_before_save \
    --pre_asr \
    --extract_whisper_text_token_by_words