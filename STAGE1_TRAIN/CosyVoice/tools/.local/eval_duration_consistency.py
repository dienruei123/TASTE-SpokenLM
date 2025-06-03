import torch
import argparse
import os
import math
import librosa
import torch.nn.functional as F
from glob import glob
from tqdm import tqdm
from pprint import pp
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, AutoModel

import librosa
import json

def parse_args():
    parser = argparse.ArgumentParser(description='calculate the emo consistency between the orig waveform and the (resynthesis) waveform.')
    parser.add_argument('--src_dir', required=True, help="where the original source waveform stores")
    parser.add_argument('--tgt_dir', required=True, help='where the target waveform to be evaluated stores')
    parser.add_argument('--tgt_subdir', required=True, help='Help with identify the exp name and eval settings')
    parser.add_argument('--output_dir', default=None, help='output_dir for the result')
    parser.add_argument('--output_fname', default='duration_consistency_result', help='output fname of the result')
    parser.add_argument('--tolerance_window', default=0.02, type=float, help="the tolerance window value when calculation hit or not.")
    args = parser.parse_args()
    pp(args)
    return args


def parse_asr_result(data_dir, file_name="asr_result.json"):
    asr_fpath = os.path.join(data_dir, file_name)
    # read json file
    with open(asr_fpath, 'r') as jfr:
        asr_result = json.load(jfr)
    # get fids
    return asr_result


def eval_duration_consistency(src_asr_result, tgt_asr_result, tolerance_window=0.02, ignore_asr_inconsistency=True):
    fid_keys = src_asr_result.keys()
    asr_utt_inconsist_count = 0
    asr_chunk_inconsist_count = 0
    total_utt_count = 0
    total_chunk_count = 0
    valid_chunk_count = 0
    hit_chunk_count = 0
    valid_duration_diff_sum = 0.0
    for fid_key in fid_keys:
        src_asr_chunks = src_asr_result[fid_key]['chunks']
        tgt_asr_chunks = tgt_asr_result[fid_key]['chunks']
        total_utt_count += 1
        total_chunk_count += len(tgt_asr_chunks)
        if len(src_asr_chunks) != len(tgt_asr_chunks):
            asr_utt_inconsist_count += 1
            asr_chunk_inconsist_count += len(tgt_asr_chunks)
            continue
        # calculate duration consistency
        for src_chunk, tgt_chunk in zip(src_asr_chunks, tgt_asr_chunks):
            if src_chunk['text'] != tgt_chunk['text']:
                asr_chunk_inconsist_count += 1
                continue
            else:
                valid_chunk_count += 1
                # compare duration
                src_duration = src_chunk['timestamp'][1] - src_chunk['timestamp'][0]
                tgt_duration = tgt_chunk['timestamp'][1] - tgt_chunk['timestamp'][0]
                duration_diff = abs(tgt_duration - src_duration)
                if duration_diff < tolerance_window:
                    hit_chunk_count += 1
                valid_duration_diff_sum += duration_diff
    # calculate 
    result = {
        'tolerance_window': tolerance_window, 
        'total_utt_count': total_utt_count,
        'total_chunk_count': total_chunk_count, 
        'asr_utt_inconsist_count': asr_utt_inconsist_count,
        'asr_chunk_inconsist_count': asr_chunk_inconsist_count,
        'valid_chunk_count': valid_chunk_count,
        'hit_chunk_count': hit_chunk_count,
        'duration_consistent_accuracy': hit_chunk_count / valid_chunk_count,
        'valid_duration_diff_sum': valid_duration_diff_sum,
        'valid_duration_diff_avg': valid_duration_diff_sum / valid_chunk_count,
    }
    return result


def main(args):

    # get asr results
    src_asr_result = parse_asr_result(args.src_dir)
    tgt_asr_result = parse_asr_result(args.tgt_dir)
    
    output_dir = args.output_dir
    output_fpath = None
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_fpath = os.path.join(output_dir, f"{args.output_fname}.json")
    
    new_result = eval_duration_consistency(src_asr_result, tgt_asr_result, tolerance_window=args.tolerance_window)
    pp(new_result)
    if output_fpath is not None:
        if os.path.exists(output_fpath):
            with open(output_fpath, 'r') as jfr:
                overall_result = json.load(jfr)
        else:
            overall_result = {}
        overall_result[args.tgt_subdir] = new_result
        with open(output_fpath, 'w') as jfw:
            json.dump(overall_result, jfw, indent=4)
        pp(f"result saved to {output_fpath}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)