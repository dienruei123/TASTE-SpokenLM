import os
import json
import torch
import argparse
import torchaudio
from tqdm import tqdm
from glob import glob
from pprint import pp


def parse_args():
    parser = argparse.ArgumentParser(description='calculate the emo consistency between the orig waveform and the (resynthesis) waveform.')
    parser.add_argument('--tgt_dir', required=True, help="where the original source waveform stores")
    parser.add_argument('--tgt_subdir', required=True, help="Help with identify the exp name and eval settings")
    parser.add_argument('--tgt_suffix', required=True, help="the file suffix to skip when parsing for the source files ids for matching.")
    parser.add_argument('--utmos_model_dir', required=True, help='the pretrained sasr_model to be used')
    parser.add_argument('--output_dir', default=None, help='output_dir for the result')
    parser.add_argument('--output_fname', default='utmos_result', help='output fname of the result')
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


def calculate_utmos_score(model_dir, fids_to_fpaths_dict, device='cuda'):
    # load model 
    utmos_predictor = torch.hub.load(f"{model_dir}", "utmos22_strong", source='local')
    utmos_predictor.to(device)
    utmos_predictor.eval()
    total = 0
    utmos_score_sum = 0.0
    with torch.inference_mode():
        for fid, fp in tqdm(fids_to_fpaths_dict.items(), total=len(fids_to_fpaths_dict)):
            wave, sr = torchaudio.load(fp)
            utmos_score_sum += utmos_predictor(wave.to(device), sr).item()
            total += 1
    utmos_result = {
        'utmos_score_sum': utmos_score_sum,
        'total': total,
        'utmos_score': utmos_score_sum / total
    }
    pp(utmos_result)
    return utmos_result


def main(args):
    fids_to_fpaths_dict = parse_dir_for_fids_to_fpaths_dict(args.tgt_dir, args.tgt_suffix)

    device = f"cuda:{args.gpu}"
    output_dir = args.output_dir
    output_fpath = None
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    output_fpath = os.path.join(output_dir, f"{args.output_fname}.json")
    
    new_result = calculate_utmos_score(args.utmos_model_dir, fids_to_fpaths_dict, device=device)    
    # save the result
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