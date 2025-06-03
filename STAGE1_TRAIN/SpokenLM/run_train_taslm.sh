#!/usr/bin/bash
# NOTE: please run this script under xxxx/emilia/taste
source ~/.bashrc
source $COSYENV

WORK_DIR=/proj/mtklmadm/dev/mtk53678
cd $WORK_DIR/rtslm/SpokenLM

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_DEBUG=INFO accelerate launch \
    --main_process_port 29500 \
    scripts/train_taslm.py \
        --config taslm/conf/baseline/0307_baseline_1B_s3-pad_wsum-fusion_drop-lm-head_text-kl_r64.yaml
        # --config taslm/conf/baseline/0307_baseline_1B_s3-pad_wsum-fusion_text-kl_r64.yaml
        # --config taslm/conf/word-delay_latent/0307_taslm_1B_eos-rvq_word-delay_wsum-fusion_drop-proj_text-kl_latent_r64_accum_grad.yaml
        # --config taslm/conf/0226_taslm_1B_eos-rvq_word-delay_gated-fusion_latent_r64.yaml
        # --config taslm/conf/0226_taslm_1B_eos-rvq_word-delay_gated-fusion_drop-proj_latent_r64.yaml
        # --config taslm/conf/0225_taslm_1B_eos-rvq_word-delay_gated-fusion_r64.yaml
        # --config taslm/conf/0226_taslm_1B_eos-rvq_word-delay_gated-fusion_drop-proj_r64.yaml
        # --config taslm/conf/0225_taslm_1B_eos-rvq_word-delay_audio-drop_skip-lm-head.yaml
        # --config taslm/conf/0213_baseline_1B_s3-pad_lora-r32.yaml
        # --config taslm/conf/0217_taslm_1B_eos-rvq_word-delay_rvq-recon.yaml
        # --config taslm/conf/0214_taslm_1B_eos-rvq_word-delay_grad-accum.yaml
        # --config taslm/conf/0213_taslm_1B_eos-rvq_word-delay.yaml
        # --config taslm/conf/0213_taslm_1B_eos-rvq_repeat.yaml