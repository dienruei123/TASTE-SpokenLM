import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import os
import shutil
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm
from cosyvoice.cli.cosyvoice import CosyVoice

from cosyvoice.utils.file_utils import read_lists, load_wav

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser(description='inference with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--audio_text_list', required=True, help='audio list for testing')
    parser.add_argument('--llm_fpath', required=True, help='llm model file')
    parser.add_argument('--flow_fpath', required=True, help='flow model file')
    parser.add_argument('--hift_fpath', required=True, help='hifigan model file')
    parser.add_argument('--model_dir', required=True, help='model dir for getting the spk embedding and speech token (possibly) in frontend')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output_dir', required=True, help='output dir for tts result')
    parser.add_argument('--delimiter', required=False, default='\t', help="delimiter for parsing audio_text_list")
    parser.add_argument('--whisper_tokenizer_dir', required=False, default="")
    parser.add_argument('--normalize_text', default=True, type=bool)
    parser.add_argument('--copy_src', action="store_true")
    parser.add_argument('--extract_target_speech_token', action="store_true")
    parser.add_argument('--adopt_teacher_forcing', action="store_true")
    parser.add_argument('--pre_asr', action="store_true")
    parser.add_argument('--pre_asr_fpath', default=None)
    parser.add_argument('--spk_emb_cutoff_threshold', type=float, default=None, help="set the threshold to only extract the first `value` seconds for spk_emb")
    parser.add_argument('--no_spk_emb', action='store_true')
    parser.add_argument('--use_target_speech_token', action='store_true', help="will ignore the TASTE tokenization and directly use s3 token. For evaluating top line only.")
    parser.add_argument('--extract_whisper_text_token_by_words', action="store_true")
    parser.add_argument('--sampling', type=int, default=25, help='sampling from top k, default k=25')
    parser.add_argument('--drop_eos_before_llm', action="store_true", help='will drop the last token after audio tower and before text encoding. Please use cauciously.')
    parser.add_argument('--normalize_and_resample_source_before_save', action="store_true", help='will load, normalize, resample the source wavform to the target sr before save for more fair generation.')

    args = parser.parse_args()
    print(args)
    return args

def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger(__name__)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Init cosyvoice models from configs
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if args.seed:
        set_seed(args.seed)

    logger.info("Loading CosyVoice model")
    cosyvoice = CosyVoice(
        args.model_dir,
        config_fpath = args.config,
        llm_fpath = args.llm_fpath,
        flow_fpath = args.flow_fpath,
        hift_fpath =args.hift_fpath,
        pre_asr=args.pre_asr,
        pre_asr_fpath=args.pre_asr_fpath,
        whisper_tokenizer_dir=args.whisper_tokenizer_dir,
    )
    logger.info("Model Loaded")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    audio_text_list = read_lists(args.audio_text_list)
    print(audio_text_list[:10])
    delimiter = args.delimiter
    with torch.inference_mode():
        # cosyvoice.model.eval() # during CosyVoiceModel.load, models are set to eval mode already
        for audio_text_line in tqdm(audio_text_list, total=len(audio_text_list), desc="Overall eval progress..."):
            audio_fpath, text = audio_text_line.split(delimiter)
            print(audio_fpath, text)
            audio_16k = load_wav(audio_fpath, target_sr=16_000)
            audio_basename = os.path.basename(audio_fpath)
            audio_name, ext_name = audio_basename.split('.')
            output_fpath = os.path.join(output_dir, f"{audio_name}_recon.wav")
            if os.path.exists(output_fpath):
                print(f"{output_fpath} exists. skip and continue")
                continue
            result = cosyvoice.inference_audio(
                text,
                audio_16k,
                audio_fpath=audio_fpath,
                normalize_text=args.normalize_text, # text normalization could be unnecessary during eval. Add an argument to toggle it. 
                extract_target_speech_token=args.extract_target_speech_token, # NOTE: Please use cautiously for teacher forcing purpose only
                adopt_teacher_forcing_for_test=args.adopt_teacher_forcing,
                extract_whisper_text_token_by_words=args.extract_whisper_text_token_by_words,
                sampling=args.sampling,
                drop_eos_before_llm=args.drop_eos_before_llm,
                spk_emb_cutoff_threshold=args.spk_emb_cutoff_threshold,
                no_spk_emb=args.no_spk_emb,
                use_target_speech_token=args.use_target_speech_token,
            )
            tts_speech = result['tts_speech']
            torchaudio.save(output_fpath, result['tts_speech'], 22050)
            src_output_fpath = os.path.join(output_dir, f"{audio_name}_orig.{ext_name}")
            if args.copy_src:
                if not args.normalize_and_resample_source_before_save:
                    shutil.copyfile(audio_fpath, src_output_fpath)
                else:
                    src_output_fpath = os.path.join(output_dir, f"{audio_name}_orig_normed.{ext_name}")
                    _target_sr = 22050 # same as tts
                    waveform, sr = torchaudio.load(audio_fpath)
                    waveform = waveform.mean(dim=0, keepdim=True)
                    if sr != _target_sr:
                        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=_target_sr)
                        waveform = resampler(waveform)
                    max_val = waveform.abs().max()
                    if max_val > 0:
                        waveform = waveform / max_val
                    torchaudio.save(src_output_fpath, waveform, _target_sr)


if __name__ == '__main__': 
    main()