#!/bin/bash
set -e
source ~/.bashrc
source $COSYENV

cd $RTSLM_WORK_DIR/CosyVoice/examples/emilia/taste

# test audio
PRETRAINED_DIR=$WORK_DIR/rtslm_storage/pretrained_models/CosyVoice-300M

EXP_ROOT=$RTSLM_WORK_DIR/CosyVoice/examples/emilia/taste/exp/llm/torch_ddp
SEED=1986

# EXP_NAME=text-only_baseline
for SAMPLING in 25 20 15 10 5; do
    for dim in 256; do
        for k in 512; do
        # EXP_NAME=0207_stg2_eos_rvq-d${dim}-l4-k${k}_sum_smaller-lr
        # EXP_NAME=0127_stg2_eos-drop_rvq-d${dim}-l4-k${k}_sum
        EXP_NAME=0127_stg2_eos-drop_rvq-d${dim}-l4-k${k}_sum_smaller-lr
        # EXP_NAME=0127_stg2_eos-drop_rvq-d${dim}-l4-k${k}_sum
        # EXP_NAME=0120_stg2_eos_rvq-d${dim}-l4-k${k}_sum
        # EXP_NAME=0120_stg2_sum_rvq-d${dim}-l4-k${k}
        # EXP_NAME=0114_stg2_sum_rvq-d${dim}-l4-k${k}
        # EXP_NAME=0110_taste-stage2_rvq-d512-l2-k512

        EXP_DIR=$EXP_ROOT/stage2/$EXP_NAME
        # EXP_DIR=$EXP_ROOT/stage2/0117A/$EXP_NAME

        # CKPT_NAME=epoch_0_step_6000
        CKPT_NAME=checkpoint_best
        # SPLIT=test-expresso
        # test_list_fpath=eval/test-audio_expresso.tsv
        SPLIT=test-clean
        test_list_fpath=eval/test-audio_${SPLIT}_shuf-small.tsv

        OUTPUT_ROOT=$RTSLM_WORK_DIR/CosyVoice/examples/emilia/taste/exp_result/stage2
        # OUTPUT_DIR=$OUTPUT_ROOT/$EXP_NAME/$SPLIT/$CKPT_NAME
        # OUTPUT_DIR=$OUTPUT_ROOT/$EXP_NAME/$SPLIT/$CKPT_NAME/teacher_forced
        # SAMPLING=20

        OUTPUT_DIR=$OUTPUT_ROOT/$EXP_NAME/$SPLIT/$CKPT_NAME/no-strip/seed-${SEED}/topk-${SAMPLING}
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
            --whisper_tokenizer_dir $WORK_DIR/rtslm_storage/pretrained_models/distil-whisper-large-v3 \
            --copy_src \
            --sampling $SAMPLING \
            --pre_asr \
            --extract_whisper_text_token_new \
            --extract_whisper_text_token_by_words 
            # --extract_target_speech_token \
            # --drop_eos_before_llm 
            # --adopt_teacher_forcing
            # --extract_whisper_text_token
        done
    done
done
# for SAMPLING in 5 10 15 20; do

#     OUTPUT_DIR=$OUTPUT_ROOT/$EXP_NAME/$SPLIT/$CKPT_NAME/topk-${SAMPLING}

#     config_fpath=$EXP_DIR/config.yaml
#     llm_fpath=$EXP_DIR/$CKPT_NAME.pt
#     flow_fpath=$PRETRAINED_DIR/flow.pt
#     hift_fpath=$PRETRAINED_DIR/hift.pt
#     python $RTSLM_WORK_DIR/CosyVoice/cosyvoice/bin/eval_audio.py \
#         --config $config_fpath \
#         --audio_text_list $test_list_fpath \
#         --llm_fpath $llm_fpath \
#         --flow_fpath $flow_fpath \
#         --hift_fpath $hift_fpath \
#         --model_dir $PRETRAINED_DIR \
#         --gpu 7 \
#         --output_dir $OUTPUT_DIR \
#         --whisper_tokenizer_dir $WORK_DIR/rtslm_storage/pretrained_models/distil-whisper-large-v3 \
#         --copy_src \
#         --sampling $SAMPLING \
#         --pre_asr \
#         --extract_target_speech_token \
#         --extract_whisper_text_token_new \
#         --extract_whisper_text_token_by_words
#         # --adopt_teacher_forcing
#         # --extract_whisper_text_token

#     echo "Finshed"
# done