#!/usr/bin/bash
# NOTE: please run this script under xxxx/emilia/taste
source ~/.bashrc
source $COSYENV

WORK_DIR=/proj/mtklmadm/dev/mtk53678
cd $WORK_DIR/rtslm/SpokenLM
GPU=0

# baseline
# exp_name=0211_baseline_1B_s3-pad
# exp_name=0307_baseline_1B_s3-pad_wsum-fusion_text-kl_r64
# taslm_pretrained_dir=$WORK_DIR/rtslm/SpokenLM/taslm/exp/$exp_name
# speech_tokenizer_pretrained_dir=/proj/mtklmadm/dev/mtk53678/rtslm_storage/pretrained_models/CosyVoice-300M # s3
# speech_decoder_pretrained_dir=/proj/mtklmadm/dev/mtk53678/rtslm_storage/pretrained_models/CosyVoice-300M
# output_root=$WORK_DIR/rtslm/SpokenLM/taslm/exp_result/$exp_name

# taslm
# exp_name=0213_taslm_1B_eos-rvq_word-delay_masked
# exp_name=0217_taslm_1B_eos-rvq_word-delay_rvq-recon_contd
# exp_name=0217_taslm_1B_eos-rvq_word-delay_rvq-recon_contd_no-accum-grad
# exp_name=0225_taslm_1B_eos-rvq_word-delay_audio-drop_skip-lm-head
# exp_name=0225_taslm_1B_eos-rvq_word-delay_gated-fusion_skip-lm-head_r64
# exp_name=0225_taslm_1B_eos-rvq_word-delay_gated-fusion_r64
# exp_name=0226_taslm_1B_eos-rvq_word-delay_gated-fusion_drop-proj_r64
# exp_name=0226_taslm_1B_eos-rvq_word-delay_gated-fusion_drop-proj_latent_r64
# exp_name=0226_taslm_1B_eos-rvq_word-delay_gated-fusion_drop-proj_latent_r64_copy
# exp_name=0226_taslm_1B_eos-rvq_word-delay_gated-fusion_latent_r64
# exp_name=0303_taslm_1B_eos-rvq_word-delay_gated-fusion_drop-proj_text-kl_latent_r64
# exp_name=0303_taslm_1B_eos-rvq_word-delay_wsum-fusion_drop-proj_text-kl_latent_r64
exp_name=0307_taslm_1B_eos-rvq_word-delay_wsum-fusion_drop-proj_text-kl_latent_r64_accum_grad
taslm_pretrained_dir=$WORK_DIR/rtslm/SpokenLM/taslm/exp/$exp_name
speech_tokenizer_pretrained_dir=$WORK_DIR/rtslm/CosyVoice/examples/emilia/taste/exp/llm/torch_ddp/stage2/0207_stg2_eos_rvq-d256-l4-k512_sum_smaller-lr # does not need to specify if it is taste tokenizer
speech_decoder_pretrained_dir=/proj/mtklmadm/dev/mtk53678/rtslm_storage/pretrained_models/CosyVoice-300M
output_root=$WORK_DIR/rtslm/SpokenLM/taslm/exp_result/$exp_name
# output_root=$WORK_DIR/rtslm/SpokenLM/taslm/exp_result/$exp_name/tts/reversed

split=test-clean_shuf-small
# split=test-clean_only-one
# split=expresso-test-normal
# split=simple-dialogue
for ckpt in 60000 50000 40000 30000 20000 best-speech-loss best-text-loss best-total-loss; do
    ckpt_name=checkpoint-$ckpt
    for text_topp in 0.9 0.8 0.7; do 
        output_dir=$output_root/cond-gen/${split}/${ckpt_name}/sampling/text_topp-${text_topp}
        CUDA_VISIBLE_DEVICES=$GPU python scripts/test_taslm_generation.py \
            --pretrained_dir $taslm_pretrained_dir \
            --ckpt_name $ckpt_name \
            --speech_tokenizer_pretrained_dir $speech_tokenizer_pretrained_dir \
            --speech_decoder_pretrained_dir $speech_decoder_pretrained_dir \
            --output_dir $output_dir \
            --text_topp $text_topp \
            --conditional_generation_fpath /proj/mtklmadm/dev/mtk53678/rtslm_storage/code/rtslm/SpokenLM/taslm/eval/test-gen_${split}.tsv \
            --num_samples 10
            # --no_latent_sampling \
            # --conditional_generation_fpath /proj/mtklmadm/dev/mtk53678/rtslm_storage/code/rtslm/SpokenLM/taslm/eval/test-gen_expresso-test-normal.tsv \
    done
done

    # --tts_text_fpath /proj/mtklmadm/dev/mtk53678/rtslm_storage/code/rtslm/SpokenLM/taslm/data/tts_example.txt \
    # --tts_text_fpath /proj/mtklmadm/dev/mtk53678/rtslm_storage/code/rtslm/SpokenLM/taslm/data/tts_example_reversed.txt \

