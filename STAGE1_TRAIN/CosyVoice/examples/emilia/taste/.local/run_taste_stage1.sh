#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
set -e
source $WORK_DIR/rtslm/path.sh

# stage1 must initialize from text-only!
stage=5
stop_stage=5

pretrained_model_dir=/proj/mtklmadm/dev/mtk53678/rtslm_storage/pretrained_models/CosyVoice-300M
# NOTE: Please prepare whisper model if needed (distil-whisper-large-v3, whisper-large-v3).
# And you need to modify and set the model path in the conf properly e.g., audio_joint_encoder_segmenter..., tokenize_whisper...
# flash_attention_2 is required!

# train llm
EXP_NAME="0120_stg1_wrap-skip-eos_word_no-vq_sum"
ckpt_fpath="/proj/mtklmadm/dev/mtk53678/rtslm_storage/code/rtslm/CosyVoice/examples/emilia/taste/exp/llm/torch_ddp/text-only_baseline/checkpoint_best.pt"
conf_fpath="/proj/mtklmadm/dev/mtk53678/rtslm_storage/code/rtslm/CosyVoice/examples/emilia/taste/conf/$EXP_NAME.yaml"

DATA_ROOT=/proj/mtklmadm/dev/mtk53678/rtslm_storage/code/rtslm/CosyVoice/examples/emilia/taste/data

export CUDA_VISIBLE_DEVICES="0,1,2,3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
_OMP_NUM_THREADS=$num_gpus
job_id=1986
dist_backend="nccl"
data_loader_num_workers=4
prefetch=128
train_engine=torch_ddp
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Run train. We only conduct llm traning for now."
  echo "Training info: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; num_gpus=$num_gpus; dataloader_num_workers=$data_loader_num_workers"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  # start training modules
  for model in llm; do
    mkdir -p `pwd`/exp/$model/$train_engine/$EXP_NAME
    cp $0 `pwd`/exp/$model/$train_engine/$EXP_NAME
    NCCL_DEBUG=INFO OMP_NUM_THREADS=$_OMP_NUM_THREADS torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
      cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config $conf_fpath \
      --train_data $DATA_ROOT/train.data.list \
      --cv_data $DATA_ROOT/dev.data.list \
      --model $model \
      --model_dir `pwd`/exp/$model/$train_engine/$EXP_NAME \
      --tensorboard_dir `pwd`/tensorboard/$model/$train_engine/$EXP_NAME \
      --ddp.dist_backend $dist_backend \
      --num_workers ${data_loader_num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --timeout 30 \
      --deepspeed_config ./conf/customized_ds.json \
      --deepspeed.save_states model+optimizer \
      --checkpoint $ckpt_fpath
  done
fi