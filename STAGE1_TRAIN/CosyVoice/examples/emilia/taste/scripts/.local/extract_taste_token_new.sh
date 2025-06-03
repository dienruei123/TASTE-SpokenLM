#!/usr/bin/bash
# NOTE: please run this script under xxxx/emilia/taste
source ~/.bashrc
source $COSYENV

WORK_DIR=/proj/mtklmadm/dev/mtk53678
cd $WORK_DIR/rtslm/CosyVoice/examples/emilia/taste
# some variables
manifest_fpath="/proj/mtklmadm/dev/mtk53678/rtslm_storage/code/rtslm/CosyVoice/examples/emilia/taste/data/data.manifest"
output_dir="/proj/gpu_d_09023_MR_dataset_augmented/emilia/en/taste_token"
# exp_root="/proj/mtklmadm/dev/mtk53678/rtslm_storage/code/rtslm/CosyVoice/examples/emilia/taste/exp/llm/torch_ddp/stage2/0117A"
exp_root="/proj/mtklmadm/dev/mtk53678/rtslm_storage/code/rtslm/CosyVoice/examples/emilia/taste/exp/llm/torch_ddp/stage2"
exp_name="0207_stg2_eos_rvq-d256-l4-k512_sum_smaller-lr"
exp_dir=$exp_root/$exp_name

start_idx=1024 # set to 0 to generate from the first line in the manifest
end_idx=-1 # set to -1 to generate to the last line in the manifest
# NOTE that the start_idx and end_idx should match in the both stages.
gpu_device_count_for_stage1=8  # use 8 gpus for raw token extraction (stage1).
cpu_device_count_for_stage2=32 # use 32 cpus for verifying and combining (stage2) for next stage training (spoken LM). 
# start preparation
mkdir -p $output_dir
echo "Processing $exp_dir, start_idx=$start_idx, end_idx=$end_idx."
# extract raw taste token
# NOTE: will use GPU
# torchrun --nproc_per_node=$gpu_device_count_for_stage1 tools/extract_taste_token_new.py \
#     --manifest_fpath $manifest_fpath \
#     --shard_size $gpu_device_count_for_stage1 \
#     --start_idx $start_idx \
#     --end_idx $end_idx \
#     --output_dir $output_dir \
#     --exp_dir $exp_dir \
#     --exp_name $exp_name \
#     --batch_size 64
# gather taste token with text, align with llm tokens.
# NOTE: cpu only! 
# NOTE: The `add_eos` and `drop_eos_before_llm` behavior will dircectly follow the config in the `exp_dir`.
python tools/extract_taste_token_new.py \
    --manifest_fpath $manifest_fpath \
    --shard_size $cpu_device_count_for_stage2 \
    --start_idx $start_idx \
    --end_idx $end_idx \
    --output_dir $output_dir \
    --exp_dir $exp_dir \
    --exp_name $exp_name \
    --verify_and_combine
