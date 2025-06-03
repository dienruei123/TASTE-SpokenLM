#!/usr/bin/bash
# NOTE: please run this script under xxxx/emilia/taste

# Preprocessing
## stage1: prepare manifest of EN split by generating manifest of arrow files
_emilia_data_dir=/proj/gpu_d_09023_MR_dataset_ARCHIVE/mtk53678/amphion___emilia-dataset
_output_dir=/proj/gpu_d_09023_MR_dataset_augmented/emilia/en
#: please ensure that _output_dir exists
# python tools/prepare_emilia_dataset.py --dir $_emilia_data_dir --output_dir $_output_dir --stage 1

## (stage2: concatenate dataset and save to disk)? -> if failed: directly extract speech token and save arrow files to disk
# python tools/prepare_emilia_dataset.py --dir $_emilia_data_dir --output_dir $_output_dir --stage 2
## stage3: extract speech token by shard
_onnx_fpath=/proj/mtklmadm/dev/mtk53678/rtslm_storage/pretrained_models/CosyVoice-300M-SFT/speech_tokenizer_v1.onnx
# python tools/prepare_emilia_dataset.py --dir $_emilia_data_dir --output_dir $_output_dir --onnx_fpath $_onnx_fpath --shard $shard --shard_size 8 --use_gpu --stage 3
# for shard in $(seq 17 1 20); do # 8065
#     python tools/prepare_emilia_dataset.py --dir $_emilia_data_dir --output_dir $_output_dir --onnx_fpath $_onnx_fpath --shard $shard --shard_size 8 --use_gpu --stage 3
# done
# for shard in $(seq 21 1 25); do # 8252
#     python tools/prepare_emilia_dataset.py --dir $_emilia_data_dir --output_dir $_output_dir --onnx_fpath $_onnx_fpath --shard $shard --shard_size 8 --use_gpu --stage 3
# done
# for shard in $(seq 4 1 10); do
#     bsub -q ML_CPU -app PyTorch -m CPU -P d_09023 -n 16 python tools/prepare_emilia_dataset.py --dir $_emilia_data_dir --output_dir $_output_dir --onnx_fpath $_onnx_fpath --shard $shard --shard_size 16 --stage 3
# done
## stage 5: collect s3_token and spk_emb npz files
# python tools/prepare_emilia_dataset.py --dir $_emilia_data_dir --output_dir $_output_dir --raw_output_dir /proj/gpu_d_09023_MR_dataset_augmented/emilia/en/s3_token/raw --target_output_dir /proj/gpu_d_09023_MR_dataset_augmented/emilia/en/s3_token/collected --stage 5
# python tools/prepare_emilia_dataset.py --dir $_emilia_data_dir --output_dir $_output_dir --raw_output_dir /proj/gpu_d_09023_MR_dataset_augmented/emilia/en/spk_emb/raw --target_output_dir /proj/gpu_d_09023_MR_dataset_augmented/emilia/en/spk_emb/collected --postfix _spk-emb.npz --stage 5
## stage 6: generate bundled arrow files
_manifest_fpath=/proj/gpu_d_09023_MR_dataset_augmented/emilia/en/manifest.tsv
_output_dir=/proj/gpu_d_09023_MR_dataset_augmented/emilia/en/arrow_for_taste
_s3_token_dir=/proj/gpu_d_09023_MR_dataset_augmented/emilia/en/s3_token/collected
_spk_emb_dir=/proj/gpu_d_09023_MR_dataset_augmented/emilia/en/spk_emb/collected
python prepare_emilia_dataset.py --dir $_emilia_data_dir --output_dir $_output_dir --manifest_fpath $_manifest_fpath --s3_token_dir $_s3_token_dir --spk_emb_dir $_spk_emb_dir --stage 6
# stage 7: generate manifest for new arrow
# python tools/prepare_emilia_dataset.py --dir $_emilia_data_dir --output_dir $_output_dir --target_output_dir $_output_dir/arrow_for_taste --stage 7

