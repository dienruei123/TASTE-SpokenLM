#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Extended from extract_speech_token.py.
# This script aims to extract the speech tokens by multi-processing, where the multi-processes counts aligns with the visible gpu amount. 

import argparse
import math
import logging
import torch
import copy
import onnxruntime
import torchaudio
import whisper
import datasets
import glob
import os
import time
import shutil
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import librosa as lb
import multiprocessing as mp
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets, load_from_disk, load_dataset
from functools import partial

DEFAULT_SHARD_SIZE=16


def extract_tokens_on_single_gpu(data):
    # parse data
    utt_wav_pairs, model_fpath, gpu_id, tqdm_desc, pbar_pos = data
    # initialize ort on single gpu
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.inter_op_num_threads = 8
    option.intra_op_num_threads = 1
    # option.intra_op_num_threads = 1
    providers = [("CUDAExecutionProvider", {'device_id': gpu_id})]
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)
    # extract speech tokens
    utt_token_pairs = []
    for utt_wav_pair in tqdm(utt_wav_pairs, desc=tqdm_desc, position=pbar_pos, total=len(utt_wav_pairs)):
        utt, wav_fpath = utt_wav_pair
        audio, sample_rate = torchaudio.load(wav_fpath)
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
        if audio.shape[1] / 16000 > 30:
            logging.warning('do not support extract speech token for audio longer than 30s')
            speech_token = []
        else:
            feat = whisper.log_mel_spectrogram(audio, n_mels=128)
            speech_token = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                                                    ort_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
        utt_token_pairs.append((utt, speech_token))
    return utt_token_pairs

def multigpu_inference(utt_wav_pairs, onnx_path, gpu_ids):
    # bucket utt_wav_pairs based on gpu device count
    bucket_len = math.ceil(len(utt_wav_pairs) / len(gpu_ids))

    data_for_mp = []
    
    for i, gpu_id in enumerate(gpu_ids):
        start = i * bucket_len
        end = min(start + bucket_len, len(utt_wav_pairs))
        utt_wav_pairs_bucket = copy.deepcopy(utt_wav_pairs[start: end])
        model_fpath = onnx_path
        tqdm_desc = f"GPU id: {gpu_id} | indices: ({start}, {end})"
        pbar_pos = i + 1
        data_for_mp.append((utt_wav_pairs_bucket, model_fpath, gpu_id, tqdm_desc, pbar_pos))
    
    with mp.Pool(processes=len(gpu_ids)) as pool:
        results = list(tqdm(pool.imap_unordered(extract_tokens_on_single_gpu, data_for_mp), position=0, total=len(gpu_ids), desc="Overall Progress..."))
    flattened_results = []
    for res in results:
        flattened_results.extend(res)
    return flattened_results


def _filter_out_split_by_key(arrow_file, target_key=""):
    _local_dataset = Dataset.from_file(arrow_file)
    if _local_dataset[0].get("__key__","").startswith("EN"):
        return (arrow_file, True)
    else:
        return (arrow_file, False)
# Stage 1: generate manifest of specific split (language)
def generate_split_by_manifest(dataset_dir, output_dir, postfix="arrow", split="EN", sort=True):
    _arrow_file_search_pattern = os.path.join(dataset_dir, "**/*.arrow")
    arrow_files = glob.glob(_arrow_file_search_pattern, recursive=True)
    print(arrow_files[:5], len(arrow_files))
    arrow_files = arrow_files

    partial_func_for_filtering = partial(_filter_out_split_by_key, target_key=split)
    
    with mp.Pool() as pool:
        filter_result = list(tqdm(
            pool.imap_unordered(partial_func_for_filtering, arrow_files),
            total=len(arrow_files),
            desc="Progress", 
        ))
    
    if sort:
        filter_result.sort(key=lambda x: x[0])
    output_fpath = os.path.join(output_dir, "manifest.tsv")
    with open(output_fpath, 'w') as fw:
        for arrow_file, is_in_split in filter_result:
            if is_in_split:
                fw.write(f"{arrow_file}\n")
    return


def concatenate_split_with_manifest_and_save_to_disk(manifest_fpath, output_dir, save_to_disk=True):
    arrow_files = []
    with open(manifest_fpath, 'r') as fr:
        for l in fr:
            arrow_files.append(l.strip())
    arrow_files = arrow_files[:10]
    datasets_list = []
    start_time = time.time()
    for arrow_file in tqdm(arrow_files, total=len(arrow_files), desc="Loading arrow files..."):
        ds = Dataset.from_file(arrow_file)
        datasets_list.append(ds)
    loaded_time = time.time()
    print(f"load collapse time: {loaded_time - start_time}")
    combined_dataset = concatenate_datasets(datasets_list)
    combined_time = time.time()
    print(f"combine collapse time: {combined_time - loaded_time}")
    for x in combined_dataset[:1000]:
        continue
    end_time = time.time()
    print(f"iterate collapse time: {end_time - combined_time}")
    print(f"load and concatenate then iter collapse time: {end_time - start_time}")
    print("="*20)
    if save_to_disk:
        combined_output_dir = os.path.join(output_dir, "combined")
        # combined_dataset.
        # os.makedirs(combined_output_dir, exist_ok=True)
        # combined_dataset.save_to_disk(combined_output_dir)
    del combined_dataset
    start_time = time.time()
    direct_from_disk_dataset = load_from_disk(combined_output_dir)
    direct_loaded_time = time.time()
    print(f"direct loaded collapse time: {direct_loaded_time - start_time}")
    for x in direct_from_disk_dataset[:1000]:
        continue
    direct_iterate_end_time = time.time()
    print(f"direct iterate collapse time: {direct_iterate_end_time - direct_loaded_time}")
    print(f"direct load then iter collapse time: {direct_iterate_end_time - start_time}")
    # try to load_datset by data_files
    dataset = load_dataset("arrow", data_files=arrow_files)
    print(dataset)



def _generate_speech_token_of_one_arrow(tuple_args, onnx_fpath="", output_dir="", shard_size=DEFAULT_SHARD_SIZE, use_gpu=False, num_gpus=0):
    try:
        idx, arrow_file = tuple_args

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.inter_op_num_threads = 1
        option.intra_op_num_threads = 1
        if use_gpu:
            gpu_id = idx % num_gpus
            providers = [("CUDAExecutionProvider", {'device_id': gpu_id})]
        else:
            providers = ["CPUExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(onnx_fpath, sess_options=option, providers=providers)
        ort_session_inputs = ort_session.get_inputs()

        output_fname = arrow_file.split('/')[-1].split('.')[0]
        _local_output_dir = os.path.join(output_dir, output_fname)
        os.makedirs(_local_output_dir, exist_ok=True)
        ds = Dataset.from_file(arrow_file)
        total_samples = len(ds)
        ds = ds.select_columns(["__key__", "mp3"])
        ds = ds.map(
            lambda x: x,
            batched=True,
            keep_in_memory=True,
        )
        key_to_speech_token_npy = {}
        pbar_position = idx%shard_size + 1
        
        for sample in tqdm(ds, total=len(ds), position=pbar_position, desc=f"progress@{pbar_position:02.0f}for{idx:05.0f}", dynamic_ncols=True):
            key = sample['__key__']
            tmp_save_fpath = os.path.join(_local_output_dir, f"{key}_s3-token.npy")
            if os.path.exists(tmp_save_fpath):
                speech_token_npy = np.load(tmp_save_fpath)
            else:
                _audio, sampling_rate = np.array(sample['mp3']['array'], dtype=np.float32), sample['mp3']['sampling_rate']
                if len(_audio) > sampling_rate * 30:
                    if len(_audio) > sampling_rate * 31:
                        assert False, f"please examine data sample, key={key}, length={len(_audio)}"
                    else:
                        with open(os.path.join(_local_output_dir, "warnings.log"), 'a') as fw:
                            _msg = f"sample {key} exceeds the 30 seconds requirements. Had trimmed for successful processing. Original length={len(_audio)}"
                            fw.write(f"{_msg}\n")
                        _audio = _audio[:sampling_rate*30]
                audio = torch.from_numpy(_audio).unsqueeze(0)
                if sampling_rate != 16000:
                    audio = torchaudio.functional.resample(audio, orig_freq=sampling_rate, new_freq=16000)
                feat = whisper.log_mel_spectrogram(audio, n_mels=128)
                speech_token_npy = ort_session.run(
                    None, 
                    {
                        ort_session_inputs[0].name: feat.detach().cpu().numpy(),
                        ort_session_inputs[1].name: np.array([feat.shape[2]], dtype=np.int32)
                    }
                )[0].flatten().astype(np.uint16)
                np.save(tmp_save_fpath, speech_token_npy)
            key_to_speech_token_npy[key] = speech_token_npy

        np.savez(f'{output_dir}/{output_fname}.npz', **key_to_speech_token_npy)
        try:
            shutil.rmtree(_local_output_dir)
        except Exception as e:
            pass
        return f"{arrow_file} SUCCESS"
    except Exception as e:
        return f"{arrow_file} FAILED, error={e}"


def generate_speech_token_by_shard(shard_index, manifest_fpath, onnx_fpath, output_dir, shard_size=DEFAULT_SHARD_SIZE, use_gpu=False, start_idx=None):
    arrow_files = []
    with open(manifest_fpath, 'r') as fr:
        for i, l in enumerate(fr):
            arrow_files.append((i, l.strip().split(' ')[0])) # add index of the arrow as part of the args for mp
    if start_idx != None:
        start_idx = start_idx
    else:
        start_idx = shard_index * shard_size
    end_idx = min(len(arrow_files), start_idx + shard_size)
    target_arrow_files = arrow_files[start_idx: end_idx]
    print(f"Processing shard {shard_index}, index from {start_idx:05.0f} to {end_idx:05.0f}")
    
    _output_dir = os.path.join(output_dir, f"{start_idx:05.0f}-{end_idx:05.0f}")
    os.makedirs(_output_dir, exist_ok=True)

    num_gpus = 0
    if use_gpu:
        assert torch.cuda.is_available(), "can not use gpu, cuda is not available!"
        num_gpus = torch.cuda.device_count()
        print(f"Will use {num_gpus} for generating speech tokens.")
    partial_function_for_generate_speech_token_of_one_arrow = partial(
        _generate_speech_token_of_one_arrow, 
        onnx_fpath=onnx_fpath, output_dir=_output_dir, shard_size=shard_size,
        use_gpu=use_gpu, num_gpus=num_gpus,
    )

    # important! 
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # will keep the parrallelization only in the mp pool of processes
    _nproc = min(shard_size, len(target_arrow_files))
    with mp.Pool(processes=_nproc) as pool:
        logs_result = list(tqdm(
            pool.imap_unordered(partial_function_for_generate_speech_token_of_one_arrow, target_arrow_files),
            position=0,
            total=len(target_arrow_files),
            desc="Overall Progress", 
            dynamic_ncols=True,
        ))
    
    log_output_fpath = os.path.join(_output_dir, "results.logs")
    failed_count = 0
    with open(log_output_fpath, 'w') as log_fw:
        for log in logs_result:
            log_fw.write(f"{log}\n")
            if "FAILED" in log:
                failed_count += 1
    print(f"Finished shard {shard_index} with shard_size={shard_size}, failed_count={failed_count}.")


def _generate_spk_emb_of_one_arrow(tuple_args, ort_session, output_dir="", proc_idx=0, cur_arrow_file_idx=0, total_arrow_file_len=1):
    try:
        idx, arrow_file = tuple_args

        ort_session_inputs = ort_session.get_inputs()
        output_fname = arrow_file.split('/')[-1].split('.')[0]
        if os.path.exists(f'{output_dir}/{output_fname}_spk-emb.npz'):
            return f"{arrow_file} SUCCESS"
        _local_output_dir = os.path.join(output_dir, output_fname)
        os.makedirs(_local_output_dir, exist_ok=True)
        ds = Dataset.from_file(arrow_file)
        total_samples = len(ds)
        ds = ds.select_columns(["__key__", "mp3"])
        ds = ds.map(
            lambda x: x,
            batched=True,
            keep_in_memory=True,
        )
        key_to_spk_emb_npy = {}
        pbar_position = proc_idx + 1
        # important! 
        # torch.set_num_threads(1)
        # torch.set_num_interop_threads(1)
        # will keep the parrallelization only in the mp pool of processes
        for sample in tqdm(ds, 
            total=len(ds), position=pbar_position, dynamic_ncols=True, 
            desc=f"progress@{pbar_position:02.0f}for{idx:05.0f}({cur_arrow_file_idx}/{total_arrow_file_len})"
        ):
            key = sample['__key__']
            tmp_save_fpath = os.path.join(_local_output_dir, f"{key}_spk-emb.npy")
            if os.path.exists(tmp_save_fpath):
                spk_emb_npy = np.load(tmp_save_fpath)
            else:
                _audio, sampling_rate = np.array(sample['mp3']['array'], dtype=np.float32), sample['mp3']['sampling_rate']
                if len(_audio) > sampling_rate * 30:
                    if len(_audio) > sampling_rate * 31:
                        assert False, f"please examine data sample, key={key}, length={len(_audio)}"
                    else:
                        with open(os.path.join(_local_output_dir, "warnings.log"), 'a') as fw:
                            _msg = f"sample {key} exceeds the 30 seconds requirements. Had trimmed for successful processing. Original length={len(_audio)}"
                            fw.write(f"{_msg}\n")
                        _audio = _audio[:sampling_rate*30]
                audio = torch.from_numpy(_audio).unsqueeze(0)
                if sampling_rate != 16000:
                    audio = torchaudio.functional.resample(audio, orig_freq=sampling_rate, new_freq=16000)
                # calculate kaldi feat
                feat = kaldi.fbank(
                    audio,
                    num_mel_bins=80,
                    dither=0,
                    sample_frequency=16000
                )
                feat = feat - feat.mean(dim=0, keepdim=True)
                feat = feat.unsqueeze(dim=0).cpu().numpy()
                spk_emb_npy = ort_session.run(
                    None, 
                    { ort_session_inputs[0].name: feat },
                )[0].flatten()
                np.save(tmp_save_fpath, spk_emb_npy)
            key_to_spk_emb_npy[key] = spk_emb_npy

        np.savez(f'{output_dir}/{output_fname}_spk-emb.npz', **key_to_spk_emb_npy)
        try:
            shutil.rmtree(_local_output_dir)
        except Exception as e:
            pass
        return f"{arrow_file} SUCCESS"
    except Exception as e:
        return f"{arrow_file} FAILED, error={e}"


def _generate_spk_emb_of_arrows(arrows_items, onnx_fpath="", output_dir="", use_gpu=False, num_gpus=0):
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.inter_op_num_threads = 1
    option.intra_op_num_threads = 1
    if use_gpu:
        gpu_id = idx % num_gpus
        providers = [("CUDAExecutionProvider", {'device_id': gpu_id})]
    else:
        providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(onnx_fpath, sess_options=option, providers=providers)

    results = []
    _total_arrow_file_len = len(arrows_items)
    for idx, arrow_item in enumerate(arrows_items):
        _bucket_id, _id, _arrow_file = arrow_item
        _result = _generate_spk_emb_of_one_arrow(
            (_id, _arrow_file), ort_session, 
            output_dir=output_dir, proc_idx=_bucket_id, cur_arrow_file_idx=idx+1, total_arrow_file_len=_total_arrow_file_len,
        )
        results.append(_result)
    return results


def generate_spk_emb_by_index(start_idx, end_idx, manifest_fpath, onnx_fpath, output_dir, nproc, use_gpu=False):
    arrow_files = []
    with open(manifest_fpath, 'r') as fr:
        for i, l in enumerate(fr):
            arrow_files.append((i, l.strip().split(' ')[0])) # add index of the arrow as part of the args for mp
    
    end_idx = min(len(arrow_files), end_idx)
    target_arrow_files = arrow_files[start_idx: end_idx]
    print(f"Processing index from {start_idx:05.0f} to {end_idx:05.0f}")
    # bucket target files for parallel
    buckets_files = [[] for _ in range(nproc)]
    for _id_arrow_file_pair in target_arrow_files:
        _id, _arrow_file = _id_arrow_file_pair
        _bucket_id = _id % nproc
        buckets_files[_bucket_id].append((_bucket_id, _id, _arrow_file))

    _output_dir = os.path.join(output_dir, f"{start_idx:05.0f}-{end_idx:05.0f}_spk-emb")
    os.makedirs(_output_dir, exist_ok=True)

    num_gpus = 0
    if use_gpu:
        assert torch.cuda.is_available(), "can not use gpu, cuda is not available!"
        num_gpus = torch.cuda.device_count()
        print(f"Will use {num_gpus} for generating speech tokens.")
    partial_function_for_generate_spk_emb_of_arrows = partial(
        _generate_spk_emb_of_arrows, 
        onnx_fpath=onnx_fpath, output_dir=_output_dir, use_gpu=use_gpu, num_gpus=num_gpus
    )

    # important! 
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # will keep the parrallelization only in the mp pool of processes
    with mp.Pool(processes=nproc) as pool:
        logs_result = list(tqdm(
            pool.imap_unordered(partial_function_for_generate_spk_emb_of_arrows, buckets_files),
            position=0,
            total=len(buckets_files),
            desc="Overall Progress", 
            dynamic_ncols=True,
        ))
    
    log_output_fpath = os.path.join(_output_dir, "results.logs")
    failed_count = 0
    with open(log_output_fpath, 'w') as log_fw:
        for buckets_log in logs_result:
            for log in buckets_log:
                log_fw.write(f"{log}\n")
                if "FAILED" in log:
                    failed_count += 1
    print(f"Finished extracting spk-emb from {start_idx}-{end_idx} with nproc={nproc}, failed_count={failed_count}.")


def collect_data(manifest_fpath, raw_dir, tgt_dir, skip_existing=True, start_idx=None, end_idx=None, postfix=".npz"):
    _search_pattern = os.path.join(raw_dir, "**/*.npz")
    print(_search_pattern)
    npz_files_to_be_collected = glob.glob(_search_pattern, recursive=True)
    print(npz_files_to_be_collected[:10], len(npz_files_to_be_collected))
    os.makedirs(tgt_dir, exist_ok=True)

    arrow_files = []
    with open(manifest_fpath, 'r') as fr:
        for i, l in enumerate(fr):
            arrow_files.append((i, l.strip().split(' ')[0])) 
    
    # create arrow name to npz_file_fpath dict
    arrow_to_npz_dict = {}
    for _npz_fpath in npz_files_to_be_collected:
        arrow_name = _npz_fpath.split('/')[-1].split(postfix)[0]
        arrow_to_npz_dict[arrow_name] = _npz_fpath
    # record remaining arrow files that does not already have .npz
    with open(os.path.join(tgt_dir, "remainings.logs"), 'w') as rfw:
        for (idx, arrow_fpath) in tqdm(arrow_files, total=len(arrow_files), desc="collecting arrow files..."):
            arrow_name = arrow_fpath.split("/")[-1].split('.')[0]
            tgt_npz_fpath = os.path.join(tgt_dir, f"{arrow_name}{postfix}")
            if os.path.exists(tgt_npz_fpath):
                continue
            src_npz_fpath = arrow_to_npz_dict.get(arrow_name, None)
            if src_npz_fpath != None:
                shutil.copy(src_npz_fpath, tgt_npz_fpath)
            else:
                rfw.write(f"{arrow_fpath} idx={idx} is not yet being processed.\n")




def _generate_one_new_arrow_with_s3_token_and_spk_emb(arrow_fpath, s3_token_fpath, spk_emb_fpath, output_dir):
    print(arrow_fpath, s3_token_fpath, spk_emb_fpath)
    ds = Dataset.from_file(arrow_fpath)
    s3_token_npz = np.load(s3_token_fpath)
    spk_emb_npz = np.load(spk_emb_fpath)
    print("Original dataset columns:", ds.column_names)

    def _orig_sample_to_new_sample(sample):
        key = sample['__key__']
        sample['s3_token'] = s3_token_npz[key]
        sample['spk_emb'] = spk_emb_npz[key]
    
        return sample
    
    new_ds = ds.map(
        _orig_sample_to_new_sample,
        keep_in_memory=True
    )

    print("Updated dataset columns:", new_ds.column_names)
    arrow_name = arrow_fpath.split('/')[-1].split('.')[0]
    new_arrow_fpath = os.path.join(output_dir, f"{arrow_name}-taste.arrow")
    save_log_fpath = os.path.join(output_dir, f"{arrow_name}-taste.log")
    res_gen = Dataset._save_to_disk_single(0, new_ds, new_arrow_fpath, None)
    with open(save_log_fpath, 'w') as fw:
        for res in res_gen:
            fw.write(f"{res}\n")
    # new_arrow_table = new_ds._data
    # ipc.write_table(new_arrow_table, new_arrow_fpath)


def generate_new_arrow_with_s3_token_and_spk_emb(manifest_fpath, s3_token_dir, spk_emb_dir, output_dir, nproc=None, start_idx=None, end_idx=None):
    arrow_files = []
    with open(manifest_fpath, 'r') as fr:
        for i, l in enumerate(fr):
            arrow_files.append((i, l.strip().split(' ')[0]))
    if start_idx != None and end_idx != None:
        print(f"Process only partial files from index {start_idx} to {end_idx}.")
        arrow_files = arrow_files[start_idx: end_idx]
    
    # arrow_files = arrow_files[:1]
    os.makedirs(output_dir, exist_ok=True)
    for (idx, _arrow_file) in tqdm(arrow_files, total=len(arrow_files), desc=f"Saving new arrows..."):
        _arrow_name = _arrow_file.split('/')[-1].split('.')[0]
        s3_token_fpath = os.path.join(s3_token_dir, f"{_arrow_name}.npz")
        spk_emb_fpath = os.path.join(spk_emb_dir, f"{_arrow_name}_spk-emb.npz")
        _generate_one_new_arrow_with_s3_token_and_spk_emb(_arrow_file, s3_token_fpath, spk_emb_fpath, output_dir)


def _calculate_metadata_of_one_arrow(arrow_file):
    ds = Dataset.from_file(arrow_file)
    total_duration = 0.0
    for x in ds:
        _duration = x['mp3']['array'].shape[0] / x['mp3']['sampling_rate'] # in secs
        total_duration += _duration
    return (arrow_file, total_duration, len(ds)) # name, duration, num_samples

def generate_manifest_for_taste_training(target_output_dir, calculate_duration=True):
    _search_pattern = os.path.join(target_output_dir, "*.arrow")
    arrow_files = glob.glob(_search_pattern)
    with mp.Pool() as pool:
        results = list(tqdm(
            pool.imap_unordered(_calculate_metadata_of_one_arrow, arrow_files),
            position=0,
            total=len(arrow_files),
            desc="Overall Progress", 
            dynamic_ncols=True,
        ))
    results.sort(key=lambda x: x[0])
    with open(os.path.join(target_output_dir, 'data.manifest'), 'w') as fw:
        for res in results:
            arrow_fpath, duration, num_samples = res
            fw.write(f"{arrow_fpath} {duration:.2f} {num_samples}\n")



def main(args):
    emilia_dataset_dir = args.dir
    output_dir = args.output_dir
    stage = args.stage

    if stage == 1:
        print("Processing stage 1...")
        generate_split_by_manifest(emilia_dataset_dir, output_dir)
        print("Finished stage 1 processing.")
    manifest_fpath = ""
    if args.manifest_fpath != None:
        manifest_fpath = args.manifest_fpath
        print(f"manifest fpath={manifest_fpath}")
    else:
        manifest_fpath = os.path.join(output_dir, "manifest.tsv")
    if stage == 2:
        print("Processing stage 2...")
        concatenate_split_with_manifest_and_save_to_disk(_manifest_fpath, output_dir)
    if stage == 3:
        print("Processing stage 3...")
        assert args.shard != None or args.start_idx != None, "Please set the shard index or start_idx!"
        assert args.onnx_fpath != "", "Please set the speech tokenizer's onnx fpath"
        generate_speech_token_by_shard(args.shard, manifest_fpath, args.onnx_fpath, output_dir, use_gpu=args.use_gpu, shard_size=args.shard_size, start_idx=args.start_idx)
    if stage == 4:
        print("Processing stage 4...")
        assert args.start_idx != None and args.end_idx != None, "Please set the start index and end index!"
        assert args.nproc != None, "Please set the n_proc!"
        assert args.onnx_fpath != "", "Please set the spk-embedding model's onnx fpath"
        generate_spk_emb_by_index(args.start_idx, args.end_idx, manifest_fpath, args.onnx_fpath, output_dir, args.nproc, use_gpu=args.use_gpu)
    if stage == 5:
        print("Processing stage 5...")
        assert args.raw_output_dir != None and args.target_output_dir != None
        collect_data(manifest_fpath, args.raw_output_dir, args.target_output_dir, postfix=args.postfix, start_idx=args.start_idx, end_idx=args.end_idx)
    if stage == 6:
        print("Processing stage 6...")
        assert args.s3_token_dir != None and args.spk_emb_dir != None
        generate_new_arrow_with_s3_token_and_spk_emb(manifest_fpath, args.s3_token_dir, args.spk_emb_dir, output_dir, nproc=args.nproc, start_idx=args.start_idx, end_idx=args.end_idx)
    if stage == 7:
        print("Processing stage 7...")
        generate_manifest_for_taste_training(args.target_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',
                        type=str)
    parser.add_argument('--manifest_fpath',
                        type=str, default=None)
    parser.add_argument('--onnx_fpath',
                        type=str, default="")
    parser.add_argument('--shard', type=int, default=None)
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--nproc', type=int, default=None)
    parser.add_argument('--shard_size', type=int, default=16)
    parser.add_argument('--use_gpu', action="store_true")
    parser.add_argument('--output_dir', type=str, help="output dir to save the extracted speech token")
    parser.add_argument('--raw_output_dir', type=str, default=None)
    parser.add_argument('--target_output_dir', type=str, default=None)
    parser.add_argument('--postfix', type=str, default=".npz")
    parser.add_argument('--s3_token_dir', type=str, default=None)
    parser.add_argument('--spk_emb_dir', type=str, default=None)
    parser.add_argument('--stage', type=int, help="Stage1: filter out specific arrow files and create manifest")
    args = parser.parse_args()
    main(args)
