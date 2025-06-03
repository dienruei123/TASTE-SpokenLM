import torch
import argparse
import os
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
    parser.add_argument('--src_suffix', required=True, help="the file suffix to skip when parsing for the source files ids for matching.")
    parser.add_argument('--tgt_suffix', required=True, help="the file suffix to skip when parsing for the target file ids for matching.")
    parser.add_argument('--ser_model_dir', required=True, help='the pretrained ser_model to be used')
    parser.add_argument('--output_dir', default=None, help='output_dir for the result')
    parser.add_argument('--output_fname', default='emo_consistency_result', help='output fname of the result')
    parser.add_argument('--topk', default=1, type=int, help="the topk value when calculation accuracy")
    parser.add_argument('--use_cosine', default=1, type=int, help="the topk value when calculation accuracy")
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


def gather_dicts(src_dict, tgt_dict):
    fids_to_src_tgt_fpaths_dict = {}
    hits = 0
    total = len(src_dict)
    # if tgt_dict covers all the fids in src_dict, then it's Okay. Otherwise raise error.
    for fid, src_fp in src_dict.items():
        tgt_fp = tgt_dict.get(fid, None)
        if tgt_fp is None:
            raise RuntimeError(f"fid={fid} is in src_dict but not found in tgt_dict. Please re-examine the data.")
        fids_to_src_tgt_fpaths_dict[fid] = {
            'src_fpath': src_fp,
            'tgt_fpath': tgt_fp
        }
        hits += 1
    print(f"Coverage from tgt_dict to src_dict={hits / total*100:3.1f}%. (Should be 100.0%)")
    return fids_to_src_tgt_fpaths_dict

def _js_divergence(src_logits, tgt_logits, epsilon=1e-12):
    p = F.softmax(src_logits, dim=-1)
    q = F.softmax(tgt_logits, dim=-1)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * torch.log((p + epsilon) / (m + epsilon)), dim=-1)
    kl_qm = torch.sum(q * torch.log((q + epsilon) / (m + epsilon)), dim=-1)

    js = 0.5 * (kl_pm + kl_qm)
    return js.item()

def eval_emo_consistency(model_dir, fids_to_src_tgt_fpaths_dict, device='cuda', topk=1):
    # load the ser processor and model
    assert topk == 1, "currently only support topk=1"
    # processor = Wav2Vec2Processor.from_pretrained(args.ser_model_dir)
    extractor = AutoFeatureExtractor.from_pretrained(args.ser_model_dir)
    model = AutoModelForAudioClassification.from_pretrained(args.ser_model_dir)
    model = model.to(device)
    # eval metric
    hits = 0
    total = 0
    js_score = 0
    id2label = model.config.id2label
    src_distribution = {id2label[i]: 0 for i in range(len(id2label))}
    tgt_distribution = {id2label[i]: 0 for i in range(len(id2label))}
    model.eval()
    with torch.inference_mode():
        for fid, src_tgt_fp_dict in tqdm(fids_to_src_tgt_fpaths_dict.items(), total=len(fids_to_src_tgt_fpaths_dict)):
            src_fp = src_tgt_fp_dict['src_fpath']
            tgt_fp = src_tgt_fp_dict['tgt_fpath']
            # eval src (orig)
            src_speech_array, sr = librosa.load(src_fp, sr=16000)
            _src_inputs = extractor(src_speech_array, sampling_rate=sr, return_tensors='pt')
            src_inputs = {key: val.to(device) if isinstance(val, torch.Tensor) else val for key, val in _src_inputs.items()}
            src_logits = model(**src_inputs).logits
            src_pred = src_logits.argmax().cpu().item()
            src_distribution[id2label[src_pred]] += 1
            # eval tgt 
            tgt_speech_array, sr = librosa.load(tgt_fp, sr=16000)
            _tgt_inputs = extractor(tgt_speech_array, sampling_rate=sr, return_tensors='pt')
            tgt_inputs = {key: val.to(device) if isinstance(val, torch.Tensor) else val for key, val in _tgt_inputs.items()}
            tgt_logits = model(**tgt_inputs).logits
            tgt_pred = tgt_logits.argmax().cpu().item()
            tgt_distribution[id2label[tgt_pred]] += 1
            # 
            if src_pred == tgt_pred:
                hits += 1
            # add js divergence
            js_score += _js_divergence(src_logits, tgt_logits)
            total += 1
    result = {
        'emo_model_dir': model_dir,
        'topk': topk,
        'hits': hits,
        'total': total,
        'accuracy': hits / total,
        'js_score_avg': js_score / total,
        'src_distribution': src_distribution,
        'tgt_distribution': tgt_distribution,
    }
    pp(result)
    return result


def main(args):

    src_fids_to_fpaths_dict = parse_dir_for_fids_to_fpaths_dict(args.src_dir, args.src_suffix)
    tgt_fids_to_fpaths_dict = parse_dir_for_fids_to_fpaths_dict(args.tgt_dir, args.tgt_suffix)
    fids_to_src_tgt_fpaths_dict = gather_dicts(src_fids_to_fpaths_dict, tgt_fids_to_fpaths_dict)
    # perform evaluation
    device = f"cuda:{args.gpu}"
    output_dir = args.output_dir
    output_fpath = None
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_fpath = os.path.join(output_dir, f"{args.output_fname}.json")
    
    new_result = eval_emo_consistency(args.ser_model_dir, fids_to_src_tgt_fpaths_dict, device=device, topk=args.topk)
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