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
src_suffix=_orig_normed.wav
# src_subdir=orig
# src_suffix=_orig.wav

for topk in 25; do
    # tgt_subdir=0207_stg2_eos_rvq-d256-l4-k512_sum_smaller-lr/checkpoint_best/seed-1986/topk-$topk
    # tgt_subdir=0208_stg2_eos_rvq-d256-l4-k512_sum_smaller-lr/checkpoint_best/seed-1986/topk-$topk
    # tgt_subdir=0127_stg2_eos-drop_rvq-d256-l4-k512_sum_smaller-lr/checkpoint_best/seed-1986/topk-$topk
    # tgt_subdir=text-only_baseline_contd/checkpoint_best/seed-1986/topk-25
    # tgt_subdir=s3_token_topline
    tgt_subdir="SpeechTokenizer/layer[0-1)"
    tgt_suffix=_recon.wav

    # ser_model_name=wav2vec2-large-superb-er
    ser_model_name=wav2vec2-lg-xlsr-en-speech-emotion-recognition-new
    ser_model_dir=/proj/mtklmadm/models/$ser_model_name

    src_dir=$root_dir/$split/$src_subdir
    tgt_dir=$root_dir/$split/$tgt_subdir

    output_dir=$root_dir/$split/emo_result/$ser_model_name

    python tools/eval_emo_consistency.py \
        --src_dir $src_dir \
        --tgt_dir $tgt_dir \
        --tgt_subdir $tgt_subdir \
        --src_suffix $src_suffix \
        --tgt_suffix $tgt_suffix \
        --ser_model_dir $ser_model_dir \
        --output_dir $output_dir \
        --topk 1 \
        --gpu 0
done

