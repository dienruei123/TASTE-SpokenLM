import os
import torch
import torchaudio
import tqdm
import logging
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from datasets import Dataset, concatenate_datasets
from collections import defaultdict
from transformers import AutoTokenizer, WhisperProcessor
from funasr.frontends.whisper_frontend import WhisperFrontend

REQUIRED_COLUMNS = [
    'speaker_embeds',
    'asr_token_ids',
    'asr_token_lengths',
    'asr_word_ids',
    'llm_token_ids',
    'llm_token_lengths',
    'llm_word_ids',
    'audio_features',
    'audio_feature_lengths',
    'speech_token_ids',
    'speech_token_lengths',
    'asr_token_alignments',
]

def process_one_sample(
    sample,
    target_sr=16_000,
    resampler_dict=None,
    whisper_feature_extractor=None,
    whisper_tokenizer=None,
    whisper_prefix_tokens=[50258, 50259, 50360, 50364], # <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
    whisper_add_eos=False, 
    strip_text=False,
    llm_tokenizer=None,
    **kwargs,
):  
    # currently only from emilia *.arrow is supported
    speech_pt = torch.tensor(sample['mp3']['array'], dtype=torch.float32)
    if speech_pt.dim() == 1:
        speech_pt = speech_pt.unsqueeze(0) # unsqueeze to match the dim of torchaudio resampling
    else:
        speech_pt = speech_pt.mean(dim=0, keepdim=True)
    orig_sr = sample['mp3']['sampling_rate']
    text = sample['json']['text']
    assert resampler_dict != None, "Please set resampler dict for faster resampling."
    resampler = resampler_dict.get((orig_sr, target_sr), None)
    if resampler == None:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        resampler_dict[(orig_sr, target_sr)] = resampler
    waveform = resampler(speech_pt)
    # extract feature
    if isinstance(whisper_feature_extractor, WhisperFrontend):
        # from cosyvoice WhisperFrontend
        audio_feats, audio_feats_len = whisper_feature_extractor(waveform, [waveform.shape[-1]])
    else:
        # from WhisperProcessor
        waveform_npy = waveform.squeeze(0).numpy()
        padded_inputs = whisper_feature_extractor([waveform_npy], sampling_rate=target_sr, return_tensors='pt')
        audio_feats = padded_inputs.input_features.transpose(1, 2) # (B, C, T) -> (B, T, C)
        audio_feats_len = torch.tensor([audio_feats.shape[1]], dtype=torch.int32)
    # TODO: revise to word-level
    if strip_text:
        text = text.strip()
    words_with_space = [wrd if i==0 else f" {wrd}" for i, wrd in enumerate(text.split(' '))]
    word_level_token_ids = whisper_tokenizer(words_with_space, add_special_tokens=False).input_ids
    asr_token_ids = []
    asr_word_ids = []
    if whisper_add_eos:
        word_level_token_ids.append([whisper_tokenizer.eos_token_id])
    for _wrd_idx, _wrd_ids in enumerate(word_level_token_ids):
        asr_token_ids.extend(_wrd_ids)
        asr_word_ids.extend([_wrd_idx] * len(_wrd_ids))
    assert len(asr_token_ids) == len(asr_word_ids), f"Something went wrong, text={text}, words_with_space={words_with_space}"
    # extract whisper text token
    _whisper_text_tokens = whisper_prefix_tokens + asr_token_ids
    whisper_text_token = torch.tensor([_whisper_text_tokens])
    whisper_text_token_len = torch.tensor([len(_whisper_text_tokens)], dtype=torch.int32)
    asr_word_ids = torch.tensor([asr_word_ids])
    # extract s3 token and spk emb
    _s3_token = sample['s3_token']
    speech_token = torch.tensor([_s3_token])
    speech_token_len = torch.tensor([len(_s3_token)], dtype=torch.int32)
    spk_embeds = F.normalize(torch.tensor([sample['spk_emb']], dtype=torch.float32), dim=1)
    skip_prefix_idx = len(whisper_prefix_tokens)
    new_sample = {
        "embedding": spk_embeds,
        "audio_feat": audio_feats,
        "audio_feat_len": audio_feats_len,
        "text_token": whisper_text_token[:,skip_prefix_idx:],
        "text_token_len": whisper_text_token_len - skip_prefix_idx,
        "whisper_text_token": whisper_text_token,
        "whisper_text_token_len": whisper_text_token_len,
        "word_ids": asr_word_ids,
        # "asr_word_ids": None,
        "speech_token": speech_token,
        "speech_token_len": speech_token_len,
        'llm_token_ids': None,
        'llm_token_lengths': None,
        'llm_word_ids': None,
        'asr_token_alignments': None,
    }
    return new_sample


def load_from_arrow(arrow_fpath, taste_token_root=None, stage="LLM", 
    whisper_processor_fpath="", llm_tokenizer_fpath="", 
    streaming=False, num_proc=64,
):
    # set taste_token_root for attaching the corresponding taste token
    ds_of_arrow = Dataset.from_file(arrow_fpath)
    if streaming:
        ds_of_arrow = ds_of_arrow.to_iterable_dataset()
    # load whisper_processor
    whisper_processor = WhisperProcessor.from_pretrained(whisper_processor_fpath)
    # set tokenizer prefix
    _forced_decoder_ids = whisper_processor.tokenizer.get_decoder_prompt_ids(
        task="transcribe",
        language='en',
        no_timestamps=True,
    )
    whisper_prefix_tokens = whisper_processor.tokenizer.prefix_tokens
    # prepare_one_sample_function
    resampler_dict = {}
    _process_one_sample = partial(
        process_one_sample, 
        resampler_dict=resampler_dict, 
        whisper_feature_extractor=whisper_processor.feature_extractor,
        whisper_tokenizer=whisper_processor.tokenizer,
        whisper_prefix_tokens=whisper_prefix_tokens
    )
    ds_processed = ds_of_arrow.map(
        _process_one_sample,
    )
    return ds_processed

def load_from_arrows(arrow_fpath_list, taste_token_root=None, stage="LLM", whisper_processor_fpath="", llm_tokenizer_fpath="", streaming=False, num_proc=64):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(2)
    ds_of_arrows = concatenate_datasets(
        [
            Dataset.from_file(_arrow_fpath) for _arrow_fpath in tqdm.tqdm(arrow_fpath_list, desc="concatenating...")
        ]
    )
    if streaming:
        ds_of_arrows = ds_of_arrows.to_iterable_dataset()
        # num_proc=1
    # load whisper_processor
    whisper_processor = WhisperProcessor.from_pretrained(whisper_processor_fpath)
    # set tokenizer prefix
    _forced_decoder_ids = whisper_processor.tokenizer.get_decoder_prompt_ids(
        task="transcribe",
        language='en',
        no_timestamps=True,
    )
    whisper_prefix_tokens = whisper_processor.tokenizer.prefix_tokens
    resampler_dict = {}
    # prepare_one_sample_function
    _process_one_sample = partial(
        process_one_sample, 
        resampler_dict=resampler_dict, 
        whisper_feature_extractor=whisper_processor.feature_extractor,
        whisper_tokenizer=whisper_processor.tokenizer, 
        whisper_prefix_tokens=whisper_prefix_tokens,
    )
    kwargs = {}
    if not streaming: 
        kwargs['num_proc'] = num_proc
    else:
        if num_proc > 1:
            logging.info(f"Cannot set num_proc > 1 when using streaming dataset!")
    ds_processed_arrows = ds_of_arrows.map(
        _process_one_sample,
        **kwargs,
    ).select_columns(
        REQUIRED_COLUMNS
    )
    return ds_processed_arrows


def pad_seq_collate_fn(batch, device=None):
    padded = {}
    # print(batch)
    for key in batch[0].keys():
        if batch[0][key] == None:
            padded[key] = None
            continue
        if isinstance(batch[0][key][0], str):
            padded[key] = [x[key] for x in batch]
            continue
        if 'len' in key or isinstance(batch[0][key][0], torch.Tensor):
            decorator = lambda x: x
        else:
            decorator = torch.tensor
        packed_list = [decorator(x[key][0]) for x in batch]
        if 'len' in key:
            padded_tensor = torch.tensor(packed_list, dtype=torch.int32)
        else:
            padded_tensor = pad_sequence(packed_list, batch_first=True, padding_value=0)

        padded[key] = padded_tensor.to(device) if device is not None else padded_tensor
    return padded

if __name__ == "__main__":
    # TEST ONE arrow
    whisper_processor_fpath = '/proj/mtklmadm/models/whisper-large-v3'
    arrow_for_test_fpath = "/proj/gpu_d_09023_MR_dataset_augmented/emilia/en/arrow_for_taste/emilia-dataset-train-02191-of-04908-taste.arrow"
    # test load ds
    ds_single_processed = load_from_arrow(arrow_for_test_fpath, whisper_processor_fpath=whisper_processor_fpath)
    # for s in tqdm.tqdm(ds_single_processed, desc="test iterate single"):
    #     logging.debug(s['audio_features'].shape)
    # results: iteration speed: about 50 iters/secs on ACP
    # TEST concatenated dataset (dev)
    # dev_split_list = "/proj/mtklmadm/dev/mtk53678/rtslm_storage/code/rtslm/CosyVoice/examples/emilia/taste/data/train.data.list"
    dev_split_list = "/proj/mtklmadm/dev/mtk53678/rtslm_storage/code/rtslm/CosyVoice/examples/emilia/taste/data/dev.data.list"
    dev_split_list = "/proj/mtklmadm/dev/mtk53678/rtslm_storage/code/rtslm/CosyVoice/examples/emilia/taste/data/for_test/test.data.list"
    arrow_file_fpaths = []
    with open(dev_split_list, 'r') as fr:
        for l in fr:
            _arrow_fpath = l.split(' ')[0]
            arrow_file_fpaths.append(_arrow_fpath)
    # ds_concatenated = concatenate_datasets(
    #     [
    #         load_from_arrow(arrow_fpath, whisper_processor_fpath=whisper_processor_fpath)
    #         for arrow_fpath in arrow_file_fpaths
    #     ]
    # )
    ds_concatenated = load_from_arrows(arrow_file_fpaths, whisper_processor_fpath=whisper_processor_fpath)
    for i, s in tqdm.tqdm(enumerate(ds_concatenated), desc="test iterate concatenated"):
        if i % 100 == 0:
            print(s.keys())
            # assert False, "stop for debug"
        # pass