import os
import glob
import argparse


source_dir = "/proj/gpu_d_09023_MR_dataset_augmented/emilia/en/taste_token/0207_stg2_eos_rvq-d256-l4-k512_sum_smaller-lr/combined_with_delayed"
target_fpath = "/proj/mtklmadm/dev/mtk53678/rtslm_storage/code/rtslm/SpokenLM/taslm/data/0207_eos/train.data.list"

search_pattern = os.path.join(source_dir, "*.arrow")
arrow_list = glob.glob(search_pattern)
arrow_list.sort()

with open(target_fpath, 'w') as fw:
    for af in arrow_list:
        fw.write(af + '\n')
