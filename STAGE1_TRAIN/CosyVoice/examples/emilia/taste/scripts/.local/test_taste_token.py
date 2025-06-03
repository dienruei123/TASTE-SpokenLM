import os
import torch
import torchaudio
import soundfile as sf
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, WhisperProcessor
from datasets import Dataset
from hyperpyyaml import load_hyperpyyaml

arrow_fpath = "/proj/gpu_d_09023_MR_dataset_augmented/emilia/en/arrow_for_taste/emilia-dataset-train-00071-of-04908-taste.arrow"
taste_token_fpath = "/proj/gpu_d_09023_MR_dataset_augmented/emilia/en/taste_token/0114_stg2_sum_rvq-d128-l4-k512/raw/00000-00256/emilia-dataset-train-00071-of-04908-taste_token.npz"

conf_fpath = "/proj/mtklmadm/dev/mtk53678/rtslm_storage/code/rtslm/CosyVoice/examples/emilia/taste/exp/llm/torch_ddp/stage2/0117A/0114_stg2_sum_rvq-d128-l4-k512/config.yaml"
ckpt_fpath = "/proj/mtklmadm/dev/mtk53678/rtslm_storage/code/rtslm/CosyVoice/examples/emilia/taste/exp/llm/torch_ddp/stage2/0117A/0114_stg2_sum_rvq-d128-l4-k512/checkpoint_best.pt"
flow_fpath = "/proj/mtklmadm/dev/mtk53678/rtslm_storage/pretrained_models/CosyVoice-300M/flow.pt"
hift_fpath = "/proj/mtklmadm/dev/mtk53678/rtslm_storage/pretrained_models/CosyVoice-300M/hift.pt"
device = "cuda:0"

with open(conf_fpath, 'r') as fr:
    config = load_hyperpyyaml(fr)

# audio llm
_llm_state_dict = torch.load(ckpt_fpath, map_location='cpu')
audio_llm = config['llm']
audio_llm.load_state_dict(_llm_state_dict, load_partial_list=[])
audio_llm.to(device)
audio_llm.eval()
# flow
flow_model = config['flow']
_flow_state_dict = torch.load(flow_fpath, map_location='cpu')
flow_model.load_state_dict(_flow_state_dict)
flow_model.to(device)
flow_model.eval()
# hift
hift_model = config['hift']
_hift_state_dict = torch.load(hift_fpath, map_location='cpu')
hift_model.load_state_dict(_hift_state_dict)
hift_model.to(device)
hift_model.eval()


ds_arrow = Dataset.from_file(arrow_fpath)
taste_token_npz = np.load(taste_token_fpath)

whisper_dir = "/proj/mtklmadm/models/whisper-large-v3"
whisper_processor = WhisperProcessor.from_pretrained(whisper_dir)
whisper_tokenizer = whisper_processor.tokenizer

output_dir = ".local"
max_samples = 5
with torch.inference_mode():
    for i, sample in enumerate(ds_arrow):
        key = sample['__key__']
        text = sample['json']['text']
        waveform = torch.tensor(sample['mp3']['array'], dtype=torch.float32)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        orig_sr = sample['mp3']['sampling_rate']
        torchaudio.save(os.path.join(output_dir, f"{key}_orig.wav"), waveform.view(1, -1), orig_sr)
        _text_token = whisper_tokenizer(text, add_special_tokens=False).input_ids
        text_token = torch.tensor([_text_token]).to(device)
        text_token_len = torch.tensor([len(_text_token)], dtype=torch.int32).to(device)
        _taste_token = taste_token_npz[key]
        taste_token = torch.tensor(np.array([_taste_token])).to(device)
        taste_token_len = torch.tensor([len(_taste_token)], dtype=torch.int32).to(device)
        spk_embed = F.normalize(torch.tensor([sample['spk_emb']], dtype=torch.float32).to(device), dim=1)
        # get llm results from inference by taste token
        tts_speech_token = audio_llm.inference(
            text=text_token, 
            text_len=text_token_len,
            prompt_text=torch.zeros(1, 0, dtype=torch.int32).to(device), 
            prompt_text_len=torch.zeros(1, dtype=torch.int32).to(device),
            prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32).to(device), 
            prompt_speech_token_len=torch.zeros(1, dtype=torch.int32).to(device),
            taste_token=taste_token,
            taste_token_len=taste_token_len,
            embedding=spk_embed,
        )
        # flow matching 
        tts_mel = flow_model.inference(
            token=tts_speech_token,
            token_len=torch.tensor([tts_speech_token.size(1)], dtype=torch.int32).to(device),
            prompt_token=torch.zeros(1, 0, dtype=torch.int32).to(device), 
            prompt_token_len=torch.zeros(1, dtype=torch.int32).to(device),
            prompt_feat=torch.zeros(1, 0, 80).to(device), 
            prompt_feat_len=torch.zeros(1, dtype=torch.int32).to(device),
            embedding=spk_embed,
        )
        # hift
        tts_speech = hift_model.inference(mel=tts_mel).cpu()
        torch.cuda.empty_cache()
        torchaudio.save(os.path.join(output_dir, f"{key}_recon.wav"), tts_speech, 22050)
        if i == max_samples: 
            break
