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
from tqdm import tqdm
import onnxruntime
import numpy as np
import torchaudio
import whisper
import multiprocessing as mp


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


def main(args):
    utt2wav = {}
    with open('{}/wav.scp'.format(args.dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]
    
    if args.gpu_ids == None:
        try:
            num_devices = torch.cuda.device_count()
            gpu_ids = list(range(num_devices))
        except Exception as e:
            print(f"Something went wrong, is cuda available? error msg={e}")
            return 1
    else:
        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(',')]
    print(f"GPU ids={gpu_ids}")


    utt2speech_token = {}
    utt_wav_pairs = list(utt2wav.items())

    utt_token_pairs = multigpu_inference(utt_wav_pairs, args.onnx_path, gpu_ids)

    for utt, speech_token in utt_token_pairs:
        utt2speech_token[utt] = speech_token

    # for utt in tqdm(utt2wav.keys()):
    #     audio, sample_rate = torchaudio.load(utt2wav[utt])
    #     if sample_rate != 16000:
    #         audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    #     if audio.shape[1] / 16000 > 30:
    #         logging.warning('do not support extract speech token for audio longer than 30s')
    #         speech_token = []
    #     else:
    #         feat = whisper.log_mel_spectrogram(audio, n_mels=128)
    #         speech_token = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
    #                                               ort_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
    #     utt2speech_token[utt] = speech_token
    
    torch.save(utt2speech_token, '{}/utt2speech_token.pt'.format(args.dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',
                        type=str)
    parser.add_argument('--onnx_path',
                        type=str)
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None
    )
    args = parser.parse_args()
    main(args)
