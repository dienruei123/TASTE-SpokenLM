#!/usr/bin/bash
# NOTE: please run this script under xxxx/emilia/taste

# Preprocessing
## stage1: prepare manifest of EN split by generating manifest of arrow files
_emilia_data_dir=/proj/gpu_d_09023_MR_dataset_ARCHIVE/mtk53678/amphion___emilia-dataset
_output_dir=/proj/gpu_d_09023_MR_dataset_augmented/emilia/en/spk_emb_prep
#: please ensure that _output_dir exists
# python tools/prepare_emilia_dataset.py --dir $_emilia_data_dir --output_dir $_output_dir --stage 1

## (stage2: concatenate dataset and save to disk)? -> if failed: directly extract speech token and save arrow files to disk
# python tools/prepare_emilia_dataset.py --dir $_emilia_data_dir --output_dir $_output_dir --stage 2
## stage3: extract speech token by shard

_onnx_fpath=/proj/mtklmadm/dev/mtk53678/rtslm_storage/pretrained_models/CosyVoice-300M-SFT/campplus.onnx
_start_idx=000
_end_idx=768
_nproc=64
python tools/prepare_emilia_dataset.py --dir $_emilia_data_dir --output_dir $_output_dir --onnx_fpath $_onnx_fpath --start_idx $_start_idx --end_idx $_end_idx --nproc $_nproc --stage 4
# for shard in $(seq 17 1 20); do # 8065
#     python tools/prepare_emilia_dataset.py --dir $_emilia_data_dir --output_dir $_output_dir --onnx_fpath $_onnx_fpath --shard $shard --shard_size 8 --use_gpu --stage 3
# done
# for shard in $(seq 21 1 25); do # 8252
#     python tools/prepare_emilia_dataset.py --dir $_emilia_data_dir --output_dir $_output_dir --onnx_fpath $_onnx_fpath --shard $shard --shard_size 8 --use_gpu --stage 3
# done
# for shard in $(seq 4 1 10); do
#     bsub -q ML_CPU -app PyTorch -m CPU -P d_09023 -n 16 python tools/prepare_emilia_dataset.py --dir $_emilia_data_dir --output_dir $_output_dir --onnx_fpath $_onnx_fpath --shard $shard --shard_size 16 --stage 3
# done
