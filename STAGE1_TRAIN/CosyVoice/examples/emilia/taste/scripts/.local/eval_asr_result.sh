#!/bin/bash
set -e
source ~/.bashrc
source $COSYENV
cd $RTSLM_WORK_DIR/CosyVoice/examples/emilia/taste

root_dir=./result_for_eval

# split=expresso-test
split=test-clean
# split=test-clean_shuf-small

ref_trans_fpath=./eval/test-audio_${split}.tsv
# src_subdir=orig_normed
# src_suffix=_orig_normed.wav
src_subdir=orig
src_suffix=_orig.wav

for topk in 10; do
    # tgt_subdir=s3_token_topline
    tgt_subdir="SpeechTokenizer/layer[0-8)"
    # tgt_subdir=orig
    # tgt_subdir=0207_stg2_eos_rvq-d256-l4-k512_sum_smaller-lr/checkpoint_best/seed-1986/topk-$topk
    # tgt_subdir=0208_stg2_eos_rvq-d256-l4-k512_sum_smaller-lr/checkpoint_best/seed-1986/topk-$topk
    # tgt_subdir=0127_stg2_eos-drop_rvq-d256-l4-k512_sum_smaller-lr/checkpoint_best/seed-1986/topk-$topk
    # tgt_subdir=text-only_baseline_contd/checkpoint_best/seed-1986/topk-25

    # tgt_suffix=_orig.wav
    tgt_suffix=_recon.wav

    asr_model_name=whisper-large-v3

    src_dir=$root_dir/$split/$src_subdir
    tgt_dir=$root_dir/$split/$tgt_subdir
    asr_model_dir=/proj/mtklmadm/models/$asr_model_name

    output_dir=$root_dir/$split/asr_result/$ser_model_name

    CUDA_VISIBLE_DEVICES=3 python tools/eval_asr.py \
        --src_dir $src_dir \
        --tgt_dir $tgt_dir \
        --src_subdir $src_subdir \
        --tgt_subdir $tgt_subdir \
        --src_suffix $src_suffix \
        --tgt_suffix $tgt_suffix \
        --asr_model_dir $asr_model_dir \
        --output_dir $output_dir \
        --ref_trans_fpath $ref_trans_fpath \
        --gpu 0
        # --eval_only \
done

