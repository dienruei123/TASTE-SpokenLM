import torch
import argparse
import os
import librosa
from functools import partial
from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm
from pprint import pp
from datasets import Dataset, Audio
from transformers import AutoFeatureExtractor, AutoModel, pipeline, AutoTokenizer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer, BasicTextNormalizer
import jiwer
import librosa
import json

def parse_args():
    parser = argparse.ArgumentParser(description='calculate the emo consistency between the orig waveform and the (resynthesis) waveform.')
    parser.add_argument('--src_dir', required=False, help="where the original source waveform stores")
    parser.add_argument('--tgt_dir', required=True, help='where the target waveform to be evaluated stores')
    parser.add_argument('--src_subdir', required=False, help='Help with identify the exp name and eval settings')
    parser.add_argument('--tgt_subdir', required=True, help='Help with identify the exp name and eval settings')
    parser.add_argument('--src_suffix', required=True, help="the file suffix to skip when parsing for the source files ids for matching.")
    parser.add_argument('--tgt_suffix', required=True, help="the file suffix to skip when parsing for the target file ids for matching.")
    parser.add_argument('--asr_model_dir', required=True, help='the pretrained sasr_model to be used')
    parser.add_argument('--ref_trans_fpath', required=False, default=None, help='the reference transcript file path. will be used for asr evaluation')
    parser.add_argument('--output_dir', default=None, help='output_dir for the result')
    parser.add_argument('--output_fname', default='asr_result', help='output fname of the result')
    parser.add_argument('--eval_only', action='store_true', help='will not load asr model for eval. calculate the result directly.')
    parser.add_argument('--gpu', default=0, type=int, help="Which device id should be used for eval.")
    args = parser.parse_args()
    pp(args)
    return args


def parse_dir_for_fids_to_fpaths_dict(data_dir, suffix_to_skip):
    _search_pattern = os.path.join(data_dir, f"*{suffix_to_skip}")
    fpaths = glob(_search_pattern, recursive=True)
    fpaths.sort(key=lambda x: x.split('/')[-1])
    print(f"Found {len(fpaths)} valid files in {_search_pattern}.")
    fids_to_fpaths = {}
    for fp in fpaths:
        fid = fp.split('/')[-1].split(suffix_to_skip)[0]
        fids_to_fpaths[fid] = fp
    return fids_to_fpaths


def gather_dicts_for_eval(ref_dict, hyp_dict):
    fids_to_ref_hyp_pairs_dict = {}
    hits = 0
    total = len(ref_dict)
    # if tgt_dict covers all the fids in src_dict, then it's Okay. Otherwise raise error.
    for fid, ref_item in ref_dict.items():
        hyp_item = hyp_dict.get(fid, None)
        if hyp_item is None:
            raise RuntimeError(f"fid={fid} is in ref_dict but not found in hyp_dict. Please re-examine the data.")
        fids_to_ref_hyp_pairs_dict[fid] = {
            'ref': ref_item,
            'hyp': hyp_item,
        }
        hits += 1
    print(f"Coverage from tgt_dict to src_dict={hits / total*100:3.1f}%. (Should be 100.0%)")
    return fids_to_ref_hyp_pairs_dict


def eval_asr(asr_pipeline, data_dir, output_dir, fids_to_fpaths_dict, data_subdir=None, batch_size=8):
    if data_subdir is None:
        output_subdir_name = data_dir.split('/')[-1]
    else:
        # output_subdir_name = data_subdir.replace('/', '__')
        output_subdir_name = data_subdir
    asr_output_dir = os.path.join(output_dir, output_subdir_name)
    os.makedirs(asr_output_dir, exist_ok=True)
    asr_output_fpath = os.path.join(asr_output_dir, 'asr_result.json')
    if os.path.exists(asr_output_fpath):
        print(f"asr_result is found at {asr_output_fpath}. will directly load and parse for eval.")
        with open(asr_output_fpath, 'r') as fr:
            asr_result = json.load(fr)
        return asr_result
    asr_result = {}
    fids = list(fids_to_fpaths_dict.keys())
    fpaths = [fids_to_fpaths_dict[k] for k in fids]
    ds = Dataset.from_dict({
        'fid': fids,
        'audio': fpaths
    })
    ds = ds.cast_column('audio', Audio(sampling_rate=16000))
    print(ds)
    print(ds[0])
    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=lambda b: b,
    )
    for batch in tqdm(dataloader, total=len(dataloader), desc=f"ASR progress of {output_subdir_name}"):
        _fids = [s['fid'] for s in batch]
        inputs = [s['audio']['array'] for s in batch]
        _asr_result = asr_pipeline(
            inputs
        )
        for fid, res in zip(_fids, _asr_result):
            asr_result[fid] = res
    
    with open(asr_output_fpath, 'w') as jfw:
        json.dump(asr_result, jfw, indent=4)
    
    return asr_result


def parse_ref_trans_for_eval(ref_trans_fpath):
    fid_to_ref_trans_dict = {}
    with open(ref_trans_fpath, 'r') as fr:
        for l in fr:
            fpath, trans = l.strip().split('\t')
            fid = fpath.split('/')[-1].split('.')[0]
            fid_to_ref_trans_dict[fid] = {
                'text': trans,
            }
    print(f"found and parsed ref trans fpath: {ref_trans_fpath}.")
    return fid_to_ref_trans_dict


def calculate_wer(fids_to_ref_hyp_pairs_dict, normalize=True):
    if normalize:
        normalizer = BasicTextNormalizer()
    else:
        normalizer = lambda x: x.strip()
    refs, hyps = [], []
    for fid, ref_hyp_pair in fids_to_ref_hyp_pairs_dict.items():
        raw_ref = ref_hyp_pair['ref']['text']
        raw_hyp = ref_hyp_pair['hyp']['text']
        normed_ref = normalizer(raw_ref)
        if normed_ref == "":
            print(f"The ref transcript is empty after normalization (fid={fid}). directly skip for evaluation")
            continue
        normed_hyp = normalizer(raw_hyp)
        refs.append(normed_ref)
        hyps.append(normed_hyp)
    
    wer = jiwer.wer(refs, hyps)
    print(f"WER Results: {wer:.3f}")
    result = {
        'wer': wer
    }
    return result


def main(args):

    tgt_fids_to_fpaths_dict = parse_dir_for_fids_to_fpaths_dict(args.tgt_dir, args.tgt_suffix)
    # perform evaluation
    device = f"cuda:{args.gpu}"
    print(f"Use device {device}")
    output_dir = args.output_dir
    output_fpath = None
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_fpath = os.path.join(output_dir, f"{args.output_fname}.json")
    if args.ref_trans_fpath is not None:
        ref_fid_to_trans = parse_ref_trans_for_eval(args.ref_trans_fpath)
    
    if not args.eval_only:
        asr_tokenizer = AutoTokenizer.from_pretrained(args.asr_model_dir)
        forced_decoder_ids = asr_tokenizer.get_decoder_prompt_ids(
            task='transcribe',
            language='en',
            no_timestamps=False
        )
        print(f"forced decoder ids: {forced_decoder_ids}")
        _asr_pipeline = pipeline(
            'automatic-speech-recognition',
            model=args.asr_model_dir,
            torch_dtype=torch.bfloat16,
            device=device,
            chunk_length_s=30,
        )
        asr_pipeline = partial(
            _asr_pipeline,
            generate_kwargs={
                'forced_decoder_ids': forced_decoder_ids,
            },
            return_timestamps='word',
        )
    else:
        asr_pipeline = None
        print("`eval_only` is specified. will not use asr model.")
    if args.ref_trans_fpath is None:
        src_fids_to_fpaths_dict = parse_dir_for_fids_to_fpaths_dict(args.src_dir, args.src_suffix)
        _src_fid_to_trans = eval_asr(asr_pipeline, args.src_dir, output_dir, src_fids_to_fpaths_dict, data_subdir=args.src_subdir)
        ref_fid_to_trans = _src_fid_to_trans
    tgt_fid_to_trans = eval_asr(asr_pipeline, args.tgt_dir, output_dir, tgt_fids_to_fpaths_dict, data_subdir=args.tgt_subdir)
    
    fids_to_ref_hyp_pairs_dict = gather_dicts_for_eval(ref_fid_to_trans, tgt_fid_to_trans)
    new_wer_result = calculate_wer(fids_to_ref_hyp_pairs_dict)

    if output_fpath is not None:
        if os.path.exists(output_fpath):
            with open(output_fpath, 'r') as jfr:
                overall_result = json.load(jfr)
        else:
            overall_result = {}
        if args.ref_trans_fpath is None:
            _ref_key = args.src_subdir
        else:
            _ref_key = args.ref_trans_fpath.split('/')[-1]
        _key = (_ref_key, args.tgt_subdir)
        overall_result[f"{_key}"] = new_wer_result
        with open(output_fpath, 'w') as jfw:
            json.dump(overall_result, jfw, indent=4)
        pp(f"result saved to {output_fpath}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)