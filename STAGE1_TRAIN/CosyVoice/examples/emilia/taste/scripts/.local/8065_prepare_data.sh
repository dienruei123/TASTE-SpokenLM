#!/usr/bin/bash
# NOTE: please run this script under xxxx/emilia/taste

# Preprocessing
## stage1: prepare manifest of EN split by generating manifest of arrow files
_emilia_data_dir=/proj/gpu_d_09023_MR_dataset_ARCHIVE/mtk53678/amphion___emilia-dataset
_output_dir=/proj/gpu_d_09023_MR_dataset_augmented/emilia/en/s3_token/raw
#: please ensure that _output_dir exists
# python tools/prepare_emilia_dataset.py --dir $_emilia_data_dir --output_dir $_output_dir --stage 1

## (stage2: concatenate dataset and save to disk)? -> if failed: directly extract speech token and save arrow files to disk
# python tools/prepare_emilia_dataset.py --dir $_emilia_data_dir --output_dir $_output_dir --stage 2
## stage3: extract speech token by shard
_onnx_fpath=/proj/mtklmadm/dev/mtk53678/rtslm_storage/pretrained_models/CosyVoice-300M-SFT/speech_tokenizer_v1.onnx
_manifest_fpath=/proj/gpu_d_09023_MR_dataset_augmented/emilia/en/manifest.tsv
for shard in $(seq 180 1 185); do # 8065
    python tools/prepare_emilia_dataset.py --dir $_emilia_data_dir --output_dir $_output_dir --manifest_fpath $_manifest_fpath --onnx_fpath $_onnx_fpath --shard $shard --shard_size 8 --use_gpu --stage 3
done
