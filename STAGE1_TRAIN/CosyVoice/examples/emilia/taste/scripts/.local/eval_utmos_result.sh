#!/bin/bash
set -e
source ~/.bashrc
source $COSYENV
cd $RTSLM_WORK_DIR/CosyVoice/examples/emilia/taste

root_dir=./result_for_eval

# split=expresso-test
split=test-clean
# split=test-clean_shuf-small

for topk in 25; do
    # tgt_subdir=0207_stg2_eos_rvq-d256-l4-k512_sum_smaller-lr/checkpoint_best/seed-1986/topk-$topk
    # tgt_subdir=0208_stg2_eos_rvq-d256-l4-k512_sum_smaller-lr/checkpoint_best/seed-1986/topk-$topk
    # tgt_subdir=0127_stg2_eos-drop_rvq-d256-l4-k512_sum_smaller-lr/checkpoint_best/seed-1986/topk-$topk
    # tgt_subdir=text-only_baseline_contd/checkpoint_best/seed-1986/topk-25
    # tgt_subdir=orig_normed
    # tgt_subdir=s3_token_topline
    tgt_subdir="SpeechTokenizer/layer[0-8)"
    tgt_suffix=_recon.wav
    # tgt_suffix=_orig_normed.wav

    utmos_model_dir=/proj/mtklmadm/dev/mtk53678/rtslm_storage/pretrained_models/SpeechMOS

    tgt_dir=$root_dir/$split/$tgt_subdir

    output_dir=$root_dir/$split/utmos_result/utmos22_strong

    python tools/eval_utmos.py \
        --tgt_dir $tgt_dir \
        --tgt_subdir $tgt_subdir \
        --tgt_suffix $tgt_suffix \
        --utmos_model_dir $utmos_model_dir \
        --output_dir $output_dir \
        --gpu 0
done

