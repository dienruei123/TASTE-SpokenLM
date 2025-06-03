# this script is about extracting s3 token from either .arrow files and .parquest files for baseline training
# TODO: 
# 1. data for `<padding>` and `sentence-level <interleaving>`
# 2. `word-level <interleaving>`

import os
import torch
import logging
import argparse
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from datasets import Dataset
from transformers import AutoTokenizer
from tools.extract_taste_token import load_manifest_and_get_arrows_list, generate_buckets_by_index_and_shard_size



def _extract_s3_token_of_one_bucket(
    bucket,
    output_dir=None,
    llm_tokenizer_fpath="/proj/mtklmadm/models/Llama-3.2-3B",
):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    if len(bucket) == 0:
        logging.info(f"No item in bucket. Will directly return.")
        return
    
    rank = bucket[0][1]
    # prepare llm tokenizer
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_fpath)

    def _arrow_orig_sample_to_new_sample(sample, arrow_fpath="", llm_tokenizer=None):
        _text = sample['json']['text'].strip()
        llm_text_token_ids = llm_tokenizer(_text, add_special_tokens=False).input_ids
        llm_text_token_ids_len = len(llm_text_token_ids)
        new_sample = {
            '__fname__': arrow_fpath,
            'text': _text,
            'llm_text_token_ids': np.array(llm_text_token_ids, dtype=np.int32),
            'llm_text_token_ids_len': llm_text_token_ids_len,
            's3_token_ids_len': len(sample['s3_token_ids']),
        }
        return new_sample
    
    for i, (idx, shard_idx, arrow_fpath) in enumerate(bucket):
        logging.info(f"[Rank {rank}] | ({i:4.0f}/{len(bucket):4.0f}) idx={idx:4.0f} | extracting s3 for baseline...", )
        if ".arrow" in arrow_fpath:
            ds = Dataset.from_file(arrow_fpath)
            ds = ds.select_columns(
                [
                    '__key__',
                    's3_token',
                    'json'
                ]
            )
            ds = ds.rename_column('s3_token',  's3_token_ids')
            _partial_arrow_orig_sample_to_new_sample = partial(
                _arrow_orig_sample_to_new_sample,
                arrow_fpath=arrow_fpath,
                llm_tokenizer=llm_tokenizer,
            )
            ds = ds.map(
                _partial_arrow_orig_sample_to_new_sample,
                keep_in_memory=True
            )
            new_ds = ds.remove_columns(['json'])

            _arrow_fname = os.path.basename(arrow_fpath).split('.')[0]
            new_arrow_fpath = os.path.join(output_dir, f"{_arrow_fname}-llm.arrow")
            save_log_fpath = os.path.join(output_dir, f"{_arrow_fname}-llm.log")
            res_gen = Dataset._save_to_disk_single(0, new_ds, new_arrow_fpath, None)
            with open(save_log_fpath, 'w') as fw:
                for res in res_gen:
                    fw.write(f"{res}\n")


def extract_s3_token_by_buckets(
    buckets_list_with_idx, 
    output_dir=None,
    mode='parallel',
    shard_size=1, 
):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _partial_function_for_extract_s3_token_by_bucket = partial(
        _extract_s3_token_of_one_bucket,
        output_dir=output_dir,
    )
    with mp.Pool(processes=shard_size) as pool:
        logs_result = list(tqdm(
            pool.imap_unordered(_partial_function_for_extract_s3_token_by_bucket, buckets_list_with_idx),
            position=0,
            total=len(buckets_list_with_idx),
            desc="Overall Progress", 
            dynamic_ncols=True,
        ))


def main(args):
    arrows_and_parquets_list = load_manifest_and_get_arrows_list(args.manifest_fpath)
    # NOTE: we want to support both .arrow and .parquet files here
    start_idx = args.start_idx
    end_idx = args.end_idx
    if end_idx == -1:
        end_idx = len(arrows_and_parquets_list)
        logging.info(f"end_idx is -1, auto set to end_idx={end_idx}")
    # generate bucket for parallel
    buckets_list_with_idx = generate_buckets_by_index_and_shard_size(arrows_and_parquets_list, args.shard_size, start_idx=start_idx, end_idx=end_idx)
    output_dir = os.path.join(args.output_root, args.mode, f"{start_idx:05.0f}-{end_idx:05.0f}")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"The output_dir will be {output_dir}.")
    if args.mode == "parallel": # just extract out the s3 tokens and the text tokens
        extract_s3_token_by_buckets(buckets_list_with_idx, output_dir=output_dir, mode=args.mode)


if __name__ == "__main__":
    # set logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_fpath', type=str, default=None)
    parser.add_argument('--shard_size', type=int, default=None)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--output_root', type=str, default=None, required=True)
    parser.add_argument('--mode', type=str, default="parallel")
    args = parser.parse_args()
    main(args)