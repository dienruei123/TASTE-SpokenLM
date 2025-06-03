#!/usr/bin/bash
# NOTE: please run this script under xxxx/emilia/taste
source ~/.bashrc
source $COSYENV

WORK_DIR=/proj/mtklmadm/dev/mtk53678
cd $WORK_DIR/rtslm/SpokenLM

manifest_fpath="/proj/mtklmadm/dev/mtk53678/rtslm_storage/code/rtslm/CosyVoice/examples/emilia/taste/data/data.manifest"
output_root="/proj/gpu_d_09023_MR_dataset_augmented/emilia/en/s3_token_for_baseline"
mode="parallel"
start_idx=0
end_idx=-1
cpu_device_count_for_preparation=64
# prepare baseline data
python scripts/prepare_baseline_data.py \
    --manifest_fpath $manifest_fpath \
    --shard_size $cpu_device_count_for_preparation \
    --start_idx $start_idx \
    --end_idx $end_idx \
    --output_root $output_root \
    --mode $mode 

