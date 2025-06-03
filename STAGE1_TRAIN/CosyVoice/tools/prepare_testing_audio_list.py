import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import os
import random

from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description='prepare audio list (audio_fpath, text) list')
    parser.add_argument('--utt2wav', required=True, help='utt id to wav fpath')
    parser.add_argument('--utt2text', required=True, help='utt id to text transcript')
    parser.add_argument('--delimiter', required=False, default=" ", help="delimiter to parse the utt2xxx files")
    parser.add_argument('--output_dir', required=True, help='result dir')
    parser.add_argument('--output_fname', required=False, default="test_audio")
    parser.add_argument('--shuffle', action="store_true", default=False)
    parser.add_argument('--seed', type=int, default=0, required=False)
    parser.add_argument('--output_delimiter', required=False, default='\t')
    args = parser.parse_args()
    print(args)
    return args

def main():
    args = get_args()
    random.seed(args.seed)
    utt2wav, utt2text = {}, {}
    delimiter = args.delimiter
    utts = []
    with open(args.utt2wav, 'r') as f:
        for l in f:
            l = l.replace('\n', '').split(delimiter)
            utt2wav[l[0]] = l[1]
            utts.append(l[0])
    with open(args.utt2text, 'r') as f:
        for l in f:
            l = l.replace('\n', '').split(delimiter)
            utt2text[l[0]] = ' '.join(l[1:])
    if args.shuffle:
        random.shuffle(utts) # shuffle directly on utts
    output_delimiter = args.output_delimiter

    os.makedirs(args.output_dir, exist_ok=True)
    output_fpath = os.path.join(args.output_dir, f"{args.output_fname}.tsv")
    with open(output_fpath, 'w') as fw:
        for utt in utts:
            wav = utt2wav[utt]
            text = utt2text[utt]
            fw.write(f"{wav}{output_delimiter}{text}\n")

    

if __name__ == '__main__':
    main()