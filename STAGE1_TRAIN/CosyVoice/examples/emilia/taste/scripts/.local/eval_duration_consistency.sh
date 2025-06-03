#!/bin/bash
set -e
source ~/.bashrc
source $COSYENV
cd $RTSLM_WORK_DIR/CosyVoice/examples/emilia/taste

root_dir=./result_for_eval

# split=expresso-test
split=test-clean
# split=test-clean_shuf-small

src_subdir=orig_normed
# src_subdir=orig
# src_suffix=_orig.wav

for topk in 10; do
    # tgt_subdir=s3_token_topline
    tgt_subdir="SpeechTokenizer/layer[0-4)"
    # tgt_subdir=0207_stg2_eos_rvq-d256-l4-k512_sum_smaller-lr/checkpoint_best/seed-1986/topk-$topk
    # tgt_subdir=0208_stg2_eos_rvq-d256-l4-k512_sum_smaller-lr/checkpoint_best/seed-1986/topk-$topk
    # tgt_subdir=0127_stg2_eos-drop_rvq-d256-l4-k512_sum_smaller-lr/checkpoint_best/seed-1986/topk-$topk
    # tgt_subdir=text-only_baseline_contd/checkpoint_best/seed-1986/topk-25

    src_dir=$root_dir/$split/asr_result/$src_subdir
    tgt_dir=$root_dir/$split/asr_result/$tgt_subdir

    tolerance_window=0.025
    output_dir=$root_dir/$split/duration_result/tolerance_window-$tolerance_window
    python tools/eval_duration_consistency.py \
        --src_dir $src_dir \
        --tgt_dir $tgt_dir \
        --tgt_subdir $tgt_subdir \
        --output_dir $output_dir \
        --tolerance_window $tolerance_window
done

