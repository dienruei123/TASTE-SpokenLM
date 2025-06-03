import os
import time
import torch
import pprint
import logging
import argparse

import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from datasets import Dataset, disable_caching
from functools import partial
from multiprocess import Pool as Second_Pool
from hyperpyyaml import load_hyperpyyaml
from transformers import AutoTokenizer, WhisperProcessor
from funasr.frontends.whisper_frontend import WhisperFrontend
from torch.utils.data import DataLoader
from tools.data_utils import process_one_sample, pad_seq_collate_fn


def load_manifest_and_get_arrows_list(manifest_fpath):
    arrows_list = []
    with open(manifest_fpath, 'r') as fr:
        for l in fr:
            _arrow_fpath = l.strip().split(' ')[0]
            arrows_list.append(_arrow_fpath)
    return arrows_list

def generate_buckets_by_index_and_shard_size(arrows_list, shard_size, start_idx=0, end_idx=-1):
    # slice arrows_list
    if end_idx != -1:
        assert end_idx > start_idx and start_idx >= 0, f"Invalid setup. end_idx should be > than start_idx, and start_idx >= 0. start_idx={start_idx}, end_idx={end_idx}"
        logging.info(f"Slice arrows_list (len={len(arrows_list)}) by [start_idx={start_idx}, end_idx={end_idx}).")
        arrows_list = arrows_list[start_idx: end_idx]
    elif start_idx > 0:
        logging.warning(f"start_idx={start_idx} > 0 but end_idx is not set. will not perform slicing.")
    # prepare bucket based on shard_size
    buckets_list_with_idx = [[] for _ in range(shard_size)]
    idx_bias = start_idx
    for i, _arrow_fpath in enumerate(arrows_list):
        idx = i + idx_bias # idx reflects the real idx in the whole arrows_list
        shard_idx = idx % shard_size
        buckets_list_with_idx[shard_idx].append((idx, shard_idx, _arrow_fpath))
    return buckets_list_with_idx

def load_from_one_arrow(
    arrow_fpath, rank=0, 
    whisper_feature_extractor=None, 
    whisper_tokenizer=None, 
    whisper_add_eos=False, 
    strip_text=False,
    streaming=False, num_proc=12,
):
    # set taste_token_root for attaching the corresponding taste token
    ds_of_arrow = Dataset.from_file(arrow_fpath)
    ds_of_arrow = ds_of_arrow.map(
        lambda x: x,
        batched=True,
        keep_in_memory=True,
    )
    if streaming:
        ds_of_arrow = ds_of_arrow.to_iterable_dataset()
    # removed_cached_files_count = ds_of_arrow.cleanup_cache_files()
    # logging.info(f"Has removed {removed_cached_files_count} cached files.")
    # 
    whisper_prefix_tokens = whisper_tokenizer.prefix_tokens
    # prepare_one_sample_function
    resampler_dict = {}
    _process_one_sample = partial(
        process_one_sample, 
        resampler_dict=resampler_dict, 
        whisper_feature_extractor=whisper_feature_extractor, 
        whisper_tokenizer=whisper_tokenizer, 
        whisper_add_eos=whisper_add_eos,
        strip_text=strip_text,
    )
    # ds map
    ds_map_kwargs = {}
    if not streaming:
        ds_map_kwargs['num_proc'] = num_proc
    
        # with Second_Pool(num_proc) as pool:
        #     ds_processed = list(tqdm(
        #         pool.imap_unordered(_process_one_sample, ds_of_arrow),
        #         position=(rank + 1),
        #         total=len(ds_of_arrow),
        #         desc=f"[Rank {rank}] | Preparing ds...", 
        #         dynamic_ncols=True,
        #     ))
    # else:
    ds_processed = ds_of_arrow.map(
        _process_one_sample,
        **ds_map_kwargs,
    )
    ds_processed = ds_processed.select_columns(
        [
            "__key__",
            "embedding",
            "audio_feat",
            "audio_feat_len",
            "text_token",
            "text_token_len",
            "whisper_text_token",
            "whisper_text_token_len",
            "word_ids",
        ]
    )
    
    return ds_processed

def extract_taste_token_by_arrows_with_shard_idx(
    bucket, # input arg that will be pass in through mp
    conf_fpath=None, ckpt_fpath=None, # for loading the model properly
    exp_dir=None, exp_name=None, output_dir=None, output_postfix=None,
    batch_size=32, dataloader_num_workers=0, prefetch_factor=None,
    ds_num_proc=8, ds_streaming=True, # kwargs for ds
    whisper_processor_fpath="/proj/mtklmadm/models/whisper-large-v3", use_cosyvoice_whisper_feature_extractor=True,
    llm_tokenizer_fpath="", # kwargs for ds
): 
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    if len(bucket) == 0:
        logging.info(f"No item in bucket. Will directly return.")
        return
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rank = bucket[0][1]
    device = f"cuda:{rank}"
    with open(conf_fpath, 'r') as fr:
        configs = load_hyperpyyaml(fr)
    # get `add_eos`
    add_eos = False
    strip_text = False
    with open(conf_fpath, 'r') as fr:
        # prevent from loading through hyperpyyaml which takes time
        for l in fr:
            if ("add_eos" in l) and ("True" in l):
                add_eos = True
                logging.info(f"`add_eos` = True in {conf_fpath}. Will also add eos when extracting taste tokens.")
            if ("strip_text" in l) and ("True" in l):
                strip_text = True
                logging.info(f"`strip_text` = True in {conf_fpath}. Will also strip text when extracting taste tokens.")

    # build audio llm
    audio_llm = configs['llm']
    _audio_llm_state_dict = torch.load(ckpt_fpath, map_location='cpu')
    audio_llm.load_state_dict(_audio_llm_state_dict, load_partial_list=[])
    taste_tokenizer = audio_llm.audio_tokenizer
    taste_tokenizer.to(device)
    taste_tokenizer.eval()
    # load whisper processor
    whisper_feature_extractor = WhisperFrontend(
        whisper_model="large-v3",
        do_pad_trim=True,
        permute=True,
    )
    _whisper_processor = WhisperProcessor.from_pretrained(whisper_processor_fpath)
    whisper_tokenizer = _whisper_processor.tokenizer
    # set tokenizer prefix
    _forced_decoder_ids = whisper_tokenizer.get_decoder_prompt_ids(
        task="transcribe",
        language='en',
        no_timestamps=True,
    )
    _pad_seq_collate_fn = partial(pad_seq_collate_fn, device=device)
    dataloader_kwargs = {
        "batch_size": batch_size, 
        # "pin_memory": True, 
        "collate_fn": _pad_seq_collate_fn,
    }
    if not ds_streaming:
        dataloader_kwargs['num_workers'] = dataloader_num_workers
        dataloader_kwargs['prefetch_factor'] = prefetch_factor
    # make output subdir
    output_dir = os.path.join(output_dir, exp_name, 'raw', output_postfix)
    os.makedirs(output_dir, exist_ok=True)
    for i, (idx, shard_idx, arrow_fpath) in enumerate(bucket):
        ds = load_from_one_arrow(
            arrow_fpath, 
            rank=rank, 
            whisper_feature_extractor=whisper_feature_extractor,
            whisper_tokenizer=whisper_tokenizer,
            whisper_add_eos=add_eos,
            strip_text=strip_text,
            streaming=ds_streaming, num_proc=ds_num_proc,
        )
        dataloader = DataLoader(
            ds,
            **dataloader_kwargs
        )
        key_to_taste_token_npy = {}
        with torch.inference_mode():
            total_num_batches = None if ds_streaming else len(dataloader)
            for batch in tqdm(
                dataloader, 
                total=total_num_batches, 
                desc=f"[Rank {rank}] | ({i:4.0f}/{len(bucket):4.0f}) idx={idx:4.0f} | extracting...", 
                dynamic_ncols=True, 
                position=shard_idx
            ):
                tokenized_results = taste_tokenizer(
                    batch['audio_feat'], batch['audio_feat'],
                    None, None, None,
                    whisper_text_token=batch['whisper_text_token'],
                    whisper_text_token_len=batch['whisper_text_token_len'],
                    words_index=None,
                    word_ids=batch['word_ids'],
                )
                quantized_indices = tokenized_results['quantized_results']['quantized_indices']
                quantized_feats_lengths = tokenized_results['quantized_results']['quantized_feats_lengths']
                logging.debug(f"quantized_indices shape={quantized_indices.shape}")
                logging.debug(f"lengths = {quantized_feats_lengths}")
                text_token = batch['text_token']
                logging.debug(f"text token shape={text_token.shape}")
                for _i, (key, tt, tl, qi, ql) in enumerate(zip(
                    batch['__key__'],
                    text_token,
                    batch['text_token_len'],
                    quantized_indices,
                    quantized_feats_lengths,
                    # batch['word_ids'],
                )):
                    # logging.info(f"key={key}")
                    tl = tl.item()
                    tt = tt[:tl].cpu().numpy()
                    ql = ql.item()
                    qi = qi[:ql, :].cpu().numpy()
                    assert tl == ql, f"text token len != quantized taste token len!2, key={key}, arrow={arrow_fpath}"
                    key_to_taste_token_npy[key] = qi
                    # wid = wid.cpu().numpy()
                    # logging.info(f"{tt}, tt shape={tt.shape}")
                    # logging.info(f"{qi}, qi shape={qi.shape}")
                    # logging.info(f"{wid[:ql]}")
                    # logging.info(f"tl={tl}")
                    # logging.info(f"ql={ql}")
            _output_fname = os.path.basename(arrow_fpath).split('.')[0]
            np.savez(os.path.join(output_dir, f"{_output_fname}_token.npz"), **key_to_taste_token_npy)
    

def extract_taste_token_by_buckets(
    buckets_list_with_idx, output_dir, output_postfix, exp_dir,
    exp_name=None, mp_num_proc=None, use_gpu=True,
    batch_size=32, dataloader_num_workers=0, prefetch_factor=None,
    **kwargs
):
    # get cosyvoice-style conf
    conf_fpath = os.path.join(exp_dir, "config.yaml")
    ckpt_fpath = os.path.join(exp_dir, "checkpoint_best.pt")
    assert os.path.exists(conf_fpath), f"Can not find {conf_fpath} in {exp_dir} by the default setup."
    assert os.path.exists(ckpt_fpath), f"Can not find {ckpt_fpath} in {exp_dir} by the default setup."
    # preparation before mp
    num_gpus = 0
    if use_gpu:
        assert torch.cuda.is_available(), "can not use gpu, cuda is not available!"
        num_gpus = torch.cuda.device_count()
        logging.info(f"Will use {num_gpus} for generating speech tokens.")
        assert num_gpus >= mp_num_proc, f"Please set mp_num_proc less or equal to num_gpus when using gpus."
    assert mp_num_proc == len(buckets_list_with_idx), f"Please set mp_num_proc equal to shard_size for complete workload distribution."
    if exp_name == None:
        logging.info(f"exp_name is None. Will automatically set to the last subfoler name of exp_dir")
        exp_name = os.path.basename(exp_dir)
        logging.info(f"-> exp_name={exp_name}")
    ## prepare partial function
    _partial_function_for_extract_taste_token_by_arrows = partial(
        extract_taste_token_by_arrows_with_shard_idx, 
        conf_fpath=conf_fpath, ckpt_fpath=ckpt_fpath, # for loading the model properly
        exp_dir=exp_dir, exp_name=exp_name, output_dir=output_dir, output_postfix=output_postfix,
        batch_size=32, dataloader_num_workers=0, prefetch_factor=None,
    )
    # run mp
    with mp.get_context('spawn').Pool(processes=mp_num_proc) as pool:
        logs_result = list(tqdm(
            pool.imap_unordered(_partial_function_for_extract_taste_token_by_arrows, buckets_list_with_idx),
            position=0,
            total=len(buckets_list_with_idx),
            desc="Overall Progress", 
            dynamic_ncols=True,
        ))

def verify_and_combine_by_arrows_with_shard_index(
    bucket, # input arg that will be pass in through mp
    conf_fpath=None, # for loading the model properly
    exp_dir=None, exp_name=None,
    output_dir=None, output_postfix=None,
    whisper_processor_fpath="/proj/mtklmadm/models/whisper-large-v3", 
    llm_tokenizer_fpath="/proj/mtklmadm/models/Llama-3.2-3B",
):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    if len(bucket) == 0:
        logging.info(f"No item in bucket. Will directly return.")
        return
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rank = bucket[0][1]
    device = f"cuda:{rank}"
    # load conf to check if add_eos is true
    add_eos = False
    strip_text = False
    drop_eos_before_llm = False
    with open(conf_fpath, 'r') as fr:
        # prevent from loading through hyperpyyaml which takes time
        for l in fr:
            if ("add_eos" in l) and ("True" in l) and (not add_eos):
                add_eos = True
                logging.info(f"`add_eos` = True in {conf_fpath}.")
            if ("strip_text" in l) and ("True" in l) and (not strip_text):
                strip_text = True
                logging.info(f"`strip_text` = True in {conf_fpath}.")
            if ("drop_eos_before_llm" in l) and ("True" in l) and (not drop_eos_before_llm):
                drop_eos_before_llm = True
                logging.info(f"`drop_eos_before_llm` = True in {conf_fpath}.")
    if add_eos:
        if drop_eos_before_llm:
            logging.info(f"`add_eos` and `drop_eos_before_llm` are both set to True. Will ignore the last rvq in taste, which is aligned with `<eos>`.")
        else:
            logging.info(f"`add_eos` is True and not `drop_eos_before_llm`. Will preserve the <eos> token and the taste token aligned with it as well.")
    elif drop_eos_before_llm:
        raise RuntimeError(f"Cannot set `drop_eos_before_llm` to True while not `add_eos` at first!")

    # load whisper processor
    whisper_processor = WhisperProcessor.from_pretrained(whisper_processor_fpath)
    # set tokenizer prefix
    _forced_decoder_ids = whisper_processor.tokenizer.get_decoder_prompt_ids(
        task="transcribe",
        language='en',
        no_timestamps=True,
    )
    asr_tokenizer = whisper_processor.tokenizer
    # load llama tokenizer
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_fpath)

    # make output subdir
    source_dir = os.path.join(output_dir, exp_name, 'raw', output_postfix)
    output_dir = os.path.join(output_dir, exp_name, 'combined', output_postfix)
    os.makedirs(output_dir, exist_ok=True)
    for i, (idx, shard_idx, arrow_fpath) in enumerate(bucket):
        ds = Dataset.from_file(arrow_fpath)
        _arrow_fname = os.path.basename(arrow_fpath).split('.')[0]
        taste_token_fpath = os.path.join(source_dir, f"{_arrow_fname}_token.npz")
        key_to_taste_token_npy = np.load(taste_token_fpath)
        key_to_new_sample = {}
        for sample in tqdm(
            ds, 
            total=len(ds), 
            desc=f"[Rank {rank}] | ({i:4.0f}/{len(bucket):4.0f}) idx={idx:4.0f} | combining...", 
            dynamic_ncols=True, 
            position=(shard_idx+1)
        ):
            key = sample['__key__']
            text = sample['json']['text']
            if strip_text:
                text = text.strip()
            _words = text.split(' ')
            words_with_space = [wrd if i==0 else f" {wrd}" for i, wrd in enumerate(text.split(' '))]
            asr_word_level_token_ids = asr_tokenizer(words_with_space, add_special_tokens=False).input_ids
            llm_word_level_token_ids = llm_tokenizer(words_with_space, add_special_tokens=False).input_ids
            if add_eos and not drop_eos_before_llm:
                # currently we're counting in eos if taste are learned with it. 
                asr_word_level_token_ids.append([asr_tokenizer.eos_token_id])
                llm_word_level_token_ids.append([llm_tokenizer.eos_token_id])
            # gather asr_token_ids and word_ids
            asr_token_ids = []
            asr_word_ids  = []
            for _asr_wrd_idx, _asr_wrd_ids in enumerate(asr_word_level_token_ids):
                asr_token_ids.extend(_asr_wrd_ids)
                asr_word_ids.extend([_asr_wrd_idx] * len(_asr_wrd_ids))
            # gather llm_token_ids and word_ids
            llm_token_ids = []
            llm_word_ids  = []
            for _llm_wrd_idx, _llm_wrd_ids in enumerate(llm_word_level_token_ids):
                llm_token_ids.extend(_llm_wrd_ids)
                llm_word_ids.extend([_llm_wrd_idx] * len(_llm_wrd_ids))
            # transform raw taste token to llm-aligned style
            _taste_token_raw = key_to_taste_token_npy[key]
            if add_eos and drop_eos_before_llm:
                taste_token_raw = _taste_token_raw[:-1] # skip the last, which is aligned with `<eos>`
            else:
                taste_token_raw = _taste_token_raw
            qsz = taste_token_raw.shape[-1]
            tsz = len(llm_word_ids)
            llm_aligned_taste_token = np.ones((tsz, qsz), dtype=np.int32) * -1 # -1 means padding in later exp
            asr_token_ids_npy = np.array(asr_token_ids)
            asr_word_ids_npy = np.array(asr_word_ids)
            prev_llm_wrd_id = -1
            for _cur_taste_idx, _llm_wrd_id in enumerate(llm_word_ids):
                if _llm_wrd_id == prev_llm_wrd_id: continue
                _pos_mask = asr_word_ids_npy == _llm_wrd_id
                # pprint.pp(_pos_mask)
                taste_token = taste_token_raw[_pos_mask][0] # take only the first one
                llm_aligned_taste_token[_cur_taste_idx] = taste_token
                prev_llm_wrd_id = _llm_wrd_id
            new_sample = {
                '__fname__': _arrow_fname,
                '__key__': key,
                '__llm__': llm_tokenizer_fpath,
                '__asr__': whisper_processor_fpath,
                'add_eos': add_eos,
                'drop_eos_before_llm': drop_eos_before_llm,
                'text': text,
                'asr_text_token_ids': np.array(asr_token_ids, dtype=np.int32),
                'asr_text_token_ids_len': len(asr_token_ids),
                'asr_word_ids': np.array(asr_word_ids, dtype=np.int32),
                'llm_text_token_ids': np.array(llm_token_ids, dtype=np.int32),
                'llm_text_token_ids_len': len(llm_token_ids),
                'llm_word_ids': np.array(llm_word_ids, dtype=np.int32),
                'raw_taste_token_ids': _taste_token_raw, # the real `raw` one
                'llm_aligned_taste_token_ids': llm_aligned_taste_token,
            }
            # pprint.pp(new_sample)
            key_to_new_sample[key] = new_sample

        ds = ds.select_columns(
            ['__key__', 'json']
        )

        def _orig_sample_to_new_sample(sample):
            key = sample['__key__']
            new_sample = key_to_new_sample[key]
            return new_sample
        
        ds = ds.map(
            _orig_sample_to_new_sample,
            keep_in_memory=True
        )
        new_ds = ds.remove_columns(['json'])
        new_arrow_fpath = os.path.join(output_dir, f"{_arrow_fname}-llm.arrow")
        save_log_fpath = os.path.join(output_dir, f"{_arrow_fname}-llm.log")
        res_gen = Dataset._save_to_disk_single(0, new_ds, new_arrow_fpath, None)
        with open(save_log_fpath, 'w') as fw:
            for res in res_gen:
                fw.write(f"{res}\n")
            

def verify_taste_token_and_combine_by_buckets(
    buckets_list_with_idx, output_dir, output_postfix, exp_dir,
    exp_name=None, mp_num_proc=None, 
):  
    # get cosyvoice-style conf
    conf_fpath = os.path.join(exp_dir, "config.yaml")
    assert os.path.exists(conf_fpath), f"Can not find {conf_fpath} in {exp_dir} by the default setup."
    # preparation before mp
    assert mp_num_proc == len(buckets_list_with_idx), f"Please set mp_num_proc equal to shard_size for complete workload distribution."
    if exp_name == None:
        logging.info(f"exp_name is None. Will automatically set to the last subfoler name of exp_dir")
        exp_name = os.path.basename(exp_dir)
        logging.info(f"-> exp_name={exp_name}")
    ## prepare partial function
    _partial_function_for_verify_and_combine_by_arrows = partial(
        verify_and_combine_by_arrows_with_shard_index, 
        conf_fpath=conf_fpath, # for loading the model properly
        exp_dir=exp_dir, exp_name=exp_name,
        output_dir=output_dir, output_postfix=output_postfix,
    )
    # run mp
    with mp.Pool(processes=mp_num_proc) as pool:
        logs_result = list(tqdm(
            pool.imap_unordered(_partial_function_for_verify_and_combine_by_arrows, buckets_list_with_idx),
            position=0,
            total=len(buckets_list_with_idx),
            desc="Overall Progress", 
            dynamic_ncols=True,
        ))



def main():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_fpath', type=str, default=None)
    parser.add_argument('--shard_size', type=int, default=None)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None, required=True)
    parser.add_argument('--exp_dir', type=str, default=None, required=True)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--verify_and_combine', action='store_true')
    args = parser.parse_args()
    # start task
    ## preprocessing
    arrows_list = load_manifest_and_get_arrows_list(args.manifest_fpath)
    if args.end_idx == -1:
        logging.info(f"end_idx is -1, auto set to end_idx={len(arrows_list)}")
        args.end_idx = len(arrows_list)
    buckets_list_with_idx = generate_buckets_by_index_and_shard_size(arrows_list, args.shard_size, start_idx=args.start_idx, end_idx=args.end_idx)
    ## main workload
    output_postfix = f"{args.start_idx:05.0f}-{args.end_idx:05.0f}"
    if not args.verify_and_combine:
        extract_taste_token_by_buckets(
            buckets_list_with_idx, args.output_dir, output_postfix, args.exp_dir,
            exp_name=args.exp_name, mp_num_proc=args.shard_size, use_gpu=True,
            batch_size=args.batch_size, dataloader_num_workers=4, prefetch_factor=32,
        )
    else:
        ## verify and combined for llm stage training
        verify_taste_token_and_combine_by_buckets(
            buckets_list_with_idx, args.output_dir, output_postfix, args.exp_dir,
            exp_name=args.exp_name, mp_num_proc=args.shard_size
        )

if __name__ == "__main__":
    # set logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    try:
        main()
    except Exception as e:
        print(e)
    finally:
        torch.cuda.empty_cache()
        