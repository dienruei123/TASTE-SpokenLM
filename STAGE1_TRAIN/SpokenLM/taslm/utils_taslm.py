import os
import glob
import yaml
import torch
import random
import numpy as np
import onnxruntime
import torch.nn.functional as F
from argparse import Namespace
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig
from functools import partial
from tqdm import tqdm
from pprint import pp
from torch.nn.utils.rnn import pad_sequence
from hyperpyyaml import load_hyperpyyaml

PAD_INDEX=-1

def generate_one_fake_data_pair(text_eos_token_id, speech_eos_token_id, text_len_range, speech_len_range):
    text_len = random.randint(*text_len_range)
    speech_len = random.randint(*speech_len_range)
    text_token_ids = torch.randint(0, 128, (text_len,))
    text_token_ids[-1] = text_eos_token_id
    speech_token_ids = torch.randint(0, 64, (speech_len,))
    speech_token_ids[-1] = speech_eos_token_id
    return {
        'text_token_ids': text_token_ids,
        'speech_token_ids': speech_token_ids,
        'text_token_length': text_len,
        'speech_token_length': speech_len,
    }

def fake_baseline_data_generator(
    text_eos_token_id, 
    speech_eos_token_id, 
    batch_size=2, 
    text_len_range=(2, 8), 
    speech_len_range=(2, 15),
    ensure_alignment=False,
):
    # currently only type: padding is supported
    text_input_ids = []
    text_input_ids_lens = []
    speech_input_ids = []
    speech_input_ids_lens = []
    for i in range(batch_size):
        data = generate_one_fake_data_pair(text_eos_token_id, speech_eos_token_id, text_len_range, speech_len_range)
        # pad the shorter one
        _text_token_ids, _text_len = data['text_token_ids'], data['text_token_length']
        _speech_token_ids, _speech_len = data['speech_token_ids'], data['speech_token_length']
        if ensure_alignment:
            # ensure alignment
            if _speech_len > _text_len:
                _text_token_ids = F.pad(_text_token_ids, (0, _speech_len - _text_len), value=text_eos_token_id)
            else:
                _speech_token_ids = F.pad(_speech_token_ids, (0, _text_len - _speech_len), value=speech_eos_token_id)
        text_input_ids.append(_text_token_ids)
        speech_input_ids.append(_speech_token_ids)
        text_input_ids_lens.append(_text_token_ids.size(0))
        speech_input_ids_lens.append(_speech_token_ids.size(0))

    bsz = batch_size
    text_speech_input_ids = text_input_ids + speech_input_ids
    padded_text_speech_labels = pad_sequence(
        text_speech_input_ids,
        batch_first=True,
        padding_value=PAD_INDEX,
    ) # labels should use padding value=-1
    padded_text_labels = padded_text_speech_labels[:bsz, :]
    padded_speech_labels = padded_text_speech_labels[bsz:, :]
    text_attention_mask_bool = (padded_text_labels != PAD_INDEX)
    speech_attention_mask_bool = (padded_speech_labels != PAD_INDEX)
    padded_text_input_ids = padded_text_labels.clone()
    padded_text_input_ids[~text_attention_mask_bool] = text_eos_token_id
    padded_speech_input_ids = padded_speech_labels.clone()
    padded_speech_input_ids[~speech_attention_mask_bool] = speech_eos_token_id
    text_attention_mask = text_attention_mask_bool.long()
    speech_attention_mask = speech_attention_mask_bool.long()

    return {
        'text_input_ids': padded_text_input_ids,
        'text_attention_mask': text_attention_mask,
        'text_labels': padded_text_labels,
        'text_input_ids_lens': text_input_ids_lens,
        'speech_input_ids': padded_speech_input_ids,
        'speech_attention_mask': speech_attention_mask,
        'speech_labels': padded_speech_labels,
        'speech_input_ids_lens': speech_input_ids_lens,
    }

def pad_seq_collate_fn(
    batch, 
    device=None,
    map_orig_eos_to_special_token_id=None, # map <|eos|> to <|reserved_special_token_i|> (the first one is 128002) to avoid using eos
    ensure_alignment=False,
    add_bos=True,
    add_eos=True,
    text_pad_idx=None,   # set to none to be the same as the eos idx
    text_bos_idx=128000,
    text_eos_idx=128001,
    speech_pad_idx=None, # set to none to be the same as the eos idx
    speech_bos_idx=4096,
    speech_eos_idx=4097,
    ignore_idx=-1,
    speech_token_column_name="s3_token_ids",
    text_token_column_name="llm_text_token_ids",
    text_token_max_len = 180, 
):
    # for each single text, speech token pairs
    text_input_ids = []
    text_input_ids_lens = []
    speech_input_ids = []
    speech_input_ids_lens = []
    for sample in batch:
        _speech_token_ids = torch.tensor(sample[speech_token_column_name])
        _text_token_ids = torch.tensor(sample[text_token_column_name])
        _speech_token_ids_len = len(_speech_token_ids)
        _text_token_ids_len = len(_text_token_ids)
        if _text_token_ids_len > text_token_max_len:
            continue
        ## if map_orig_eos_to_special_token_id is set and found eos token in input, map the eos token to the assigned special token
        if map_orig_eos_to_special_token_id:
            _mask_for_mapping = _text_token_ids == text_eos_idx
            _text_token_ids[_mask_for_mapping] = map_orig_eos_to_special_token_id
        ## if add bos or add eos, add bos and eos (both modality)
        if add_bos:
            _text_token_ids = F.pad(_text_token_ids, (1, 0), value=text_bos_idx)
            _text_token_ids_len += 1
            _speech_token_ids = F.pad(_speech_token_ids, (1, 0), value=speech_bos_idx)
            _speech_token_ids_len += 1
        if add_eos:
            _text_token_ids = F.pad(_text_token_ids, (0, 1), value=text_eos_idx)
            _text_token_ids_len += 1
            _speech_token_ids = F.pad(_speech_token_ids, (0, 1), value=speech_eos_idx)
            _speech_token_ids_len += 1
        ## if ensure alignment: pad the short one to the longer one (with pad idx)
        if text_pad_idx == None:
            text_pad_idx = text_eos_idx
        if speech_pad_idx == None:
            speech_pad_idx = speech_eos_idx
        if ensure_alignment:
            if _speech_token_ids_len > _text_token_ids_len:
                ### pad text to speech length
                _text_token_ids = F.pad(_text_token_ids, (0, _speech_token_ids_len - _text_token_ids_len), value=text_pad_idx)
                _text_token_ids_len = _speech_token_ids_len
            else:
                _speech_token_ids = F.pad(_speech_token_ids, (0, _text_token_ids_len - _speech_token_ids_len), value=speech_pad_idx)
                _speech_token_ids_len = _text_token_ids_len
                ### pad speech to text length
        ## add to list
        text_input_ids.append(_text_token_ids)
        text_input_ids_lens.append(_text_token_ids_len)
        speech_input_ids.append(_speech_token_ids)
        speech_input_ids_lens.append(_speech_token_ids_len)
    # pad with ignore index for the labels of the whole batch
    bsz = len(text_input_ids)
    text_speech_input_ids = text_input_ids + speech_input_ids
    # print(text_speech_input_ids)
    padded_text_speech_labels = pad_sequence(
        text_speech_input_ids,
        batch_first=True,
        padding_value=ignore_idx,
    ) # labels should use padding value=-1
    padded_text_labels = padded_text_speech_labels[:bsz, :]
    padded_speech_labels = padded_text_speech_labels[bsz:, :]
    # print(padded_text_labels, padded_text_labels.shape)
    # print(padded_speech_labels, padded_speech_labels.shape)
    # generate attention masks of the padded input (by inspecting the ignore index)
    text_attention_mask_bool = (padded_text_labels != ignore_idx)
    speech_attention_mask_bool = (padded_speech_labels != ignore_idx)
    # print(text_attention_mask_bool)
    # print(speech_attention_mask_bool)
    # print((text_attention_mask_bool == speech_attention_mask_bool).sum())
    # clone the labels and generate input tokens by transforming ignore_idx to pad_idx
    padded_text_input_ids = padded_text_labels.clone()
    padded_text_input_ids[~text_attention_mask_bool] = text_eos_idx
    padded_speech_input_ids = padded_speech_labels.clone()
    padded_speech_input_ids[~speech_attention_mask_bool] = speech_eos_idx
    # print(padded_text_input_ids)
    # print(padded_speech_input_ids)
    # change attention mask to proper dtype
    text_attention_mask = text_attention_mask_bool.long()
    speech_attention_mask = speech_attention_mask_bool.long()
    padded_inputs = {
        'text_input_ids': padded_text_input_ids,
        'text_attention_mask': text_attention_mask,
        'text_labels': padded_text_labels,
        'text_input_ids_lens': torch.tensor(text_input_ids_lens),
        'speech_input_ids': padded_speech_input_ids,
        'speech_attention_mask': speech_attention_mask,
        'speech_labels': padded_speech_labels,
        'speech_input_ids_lens': torch.tensor(speech_input_ids_lens),
    }
    if device != None:
        for key, val in padded_inputs.items():
            padded_inputs[key] = val.to(device)

    return padded_inputs


def pad_seq_collate_fn_for_taste(
    batch, 
    device=None,
    map_orig_eos_to_special_token_id=None, # map <|eos|> to <|reserved_special_token_i|> (the first one is 128002) to avoid using eos
    ensure_alignment=False,
    force_recompute_delayed=True,
    use_delayed_token=True,
    add_bos=True,
    add_eos=True,
    text_pad_idx=None,   # set to none to be the same as the eos idx
    text_bos_idx=128000,
    text_eos_idx=128001,
    speech_pad_idx=None, # set to none to be the same as the eos idx
    speech_bos_idx=4096,
    speech_eos_idx=4097,
    ignore_idx=-1,
    speech_token_column_name="llm_aligned_taste_token_ids",
    text_token_column_name="llm_text_token_ids",
    text_token_max_len = 150, # 30 secs * ~5 tok / sec is set to be maximum -> filter out some hallucinations
):
    # for each single text, speech token pairs
    text_input_ids = []
    text_label_ids = []
    text_input_ids_lens = []
    speech_input_ids = []
    speech_label_ids = []
    speech_label_ids_mask = []
    speech_input_ids_lens = []
    if text_pad_idx == None:
        text_pad_idx = text_eos_idx
    if speech_pad_idx == None:
        speech_pad_idx = speech_eos_idx
    for sample in batch:
        # handle text_input
        _text_token_ids = torch.tensor(sample[text_token_column_name])
        if len(_text_token_ids) > text_token_max_len:
            print(f"filter out text token, len={len(_text_token_ids)}, ids={_text_token_ids}")
            continue
        ## if map_orig_eos_to_special_token_id is set and found eos token in input, map the eos token to the assigned special token
        if map_orig_eos_to_special_token_id:
            _special_token = map_orig_eos_to_special_token_id
            _map_mask = _text_token_ids == text_eos_idx
            _text_token_ids[_map_mask] = _special_token
        if add_bos:
            _text_token_ids = torch.cat([torch.tensor([text_bos_idx]), _text_token_ids])
        if add_eos:
            _text_token_ids = torch.cat([_text_token_ids, torch.tensor([text_eos_idx])])
        _text_token_ids_len = len(_text_token_ids)
        # handle text label
        _text_label_ids = _text_token_ids.clone() # the same as token ids. Note that during forward it will be shifted by 1.
        # _text_label_ids[-1] = ignore_idx
        # handle speech_input -> word-level token, need to handle delayed-by-word
        ## the aligned ones are for labels. <- need more consideration
        _aligned_speech_token_ids = torch.tensor(sample[speech_token_column_name]) # the `aligned` ones. should be used to be the labels
        _tsz, _qsz = _aligned_speech_token_ids.shape
        ## if add bos or add eos, add bos and eos (both modality)
        ## now handle the input. should be the word-level delayed ones.
        if sample.get('llm_delayed_taste_token_ids', None) is None or force_recompute_delayed:
            _llm_word_ids = np.array(sample['llm_word_ids'])
            _word_id_to_taste_token_dict = {}
            _prev_llm_wrd_id = min(_llm_word_ids) - 1 # _asr_wrd_id might start from 1, but it's okay. we just want the first initial wrd is different from the prev one. 
            _word_id_to_taste_token_dict[_prev_llm_wrd_id] = np.ones(_qsz, dtype=np.int32) * speech_pad_idx
            for _llm_wrd_id, _taste_token in zip(_llm_word_ids, _aligned_speech_token_ids):
                if _llm_wrd_id == _prev_llm_wrd_id: continue # word_id=_asr_wrd_id already has taste token registered
                _word_id_to_taste_token_dict[_llm_wrd_id] = _taste_token
                _prev_llm_wrd_id = _llm_wrd_id
            _delayed_speech_token_ids = np.zeros((_tsz, _qsz), dtype=np.int32)
            for _cur_idx, _prev_llm_wrd_id in enumerate(_llm_word_ids - 1):
                _prev_taste_token = _word_id_to_taste_token_dict[_prev_llm_wrd_id]
                _delayed_speech_token_ids[_cur_idx] = _prev_taste_token
        else:
            _delayed_speech_token_ids = sample['llm_delayed_taste_token_ids']
        _delayed_speech_token_ids = torch.tensor(_delayed_speech_token_ids)
        # add the last token to the _delayed ones if use_delayed token as input and output
        if use_delayed_token:
            _delayed_speech_token_ids = torch.cat([_delayed_speech_token_ids, _aligned_speech_token_ids[-1:]]) 
            # for _dl_id, _al_id, _llm_wrd_id in zip(_delayed_speech_token_ids[:-1], _aligned_speech_token_ids, sample['llm_word_ids']):
            #     print(_dl_id, _al_id, _llm_wrd_id)
            # assert False, "stop"
        
        _speech_token_ids = _delayed_speech_token_ids
        
        if not use_delayed_token:
            # raise NotImplementedError(f"using aligned_speech token is deprecated.")
            _speech_label_ids = _aligned_speech_token_ids.clone() # keep the aligned ones aligned for later usage
        else:
            _speech_label_ids = _delayed_speech_token_ids.clone()
        
        if add_bos:
            # NOTE: we want to additionally shift the speech token label by one, so that the end of the taste token is at the next begging of text word
            if not use_delayed_token:
                _prefix_pad_for_speech_label_ids = torch.ones((2, _qsz), dtype=_speech_label_ids.dtype) * speech_bos_idx
                _speech_label_ids = torch.cat([_prefix_pad_for_speech_label_ids, _speech_label_ids])
                _llm_word_ends_mask = torch.tensor(sample['llm_word_ends_mask'], dtype=torch.bool)
                _pad_for_word_mask = torch.ones(2, dtype=torch.bool)
                _speech_label_ids_mask = torch.cat([_pad_for_word_mask, _llm_word_ends_mask]) # use this for eval purpose
            else:
                _prefix_pad_for_speech_label_ids = torch.ones((1, _qsz), dtype=_speech_label_ids.dtype) * speech_bos_idx
                _speech_label_ids = torch.cat([_prefix_pad_for_speech_label_ids, _speech_label_ids])
                _llm_word_starts_mask = torch.tensor(sample['llm_word_starts_mask'], dtype=torch.bool)
                _pad_for_word_mask = torch.ones(1, dtype=torch.bool)
                _speech_label_ids_mask = torch.cat([_pad_for_word_mask, _llm_word_starts_mask, _pad_for_word_mask]) # use this for eval purpose
            _bos_pad_for_speech_token_ids = torch.ones((1, _qsz), dtype=_speech_label_ids.dtype) * speech_bos_idx
            _speech_token_ids = torch.concat([_bos_pad_for_speech_token_ids, _speech_token_ids])
        if add_eos:
            if not use_delayed_token:
                _eos_pad_for_speech_token_ids = torch.ones((1, _qsz), dtype=_speech_label_ids.dtype) * speech_eos_idx
                _speech_token_ids = torch.concat([_speech_token_ids, _eos_pad_for_speech_token_ids])
        _speech_token_ids_len = len(_speech_token_ids)

        ## if ensure alignment: pad the short one to the longer one (with pad idx)
        if ensure_alignment:
            assert _speech_token_ids_len == _text_token_ids_len
            ### pad speech to text length
        ## add to list
        text_input_ids.append(_text_token_ids)
        text_label_ids.append(_text_label_ids)
        text_input_ids_lens.append(_text_token_ids_len)
        speech_input_ids.append(_speech_token_ids)
        speech_label_ids.append(_speech_label_ids)
        speech_label_ids_mask.append(_speech_label_ids_mask)
        speech_input_ids_lens.append(_speech_token_ids_len)
        # for _taste_token, _text_token, is_word_start in zip(_speech_token_ids, _text_token_ids, _speech_label_ids_mask):
        #     print(_taste_token, _text_token, is_word_start)
        # pp(_text_token_ids)
        # pp(_text_label_ids)
        # pp(_llm_word_ids)
        # pp(_speech_label_ids_mask)
        # pp(_speech_label_ids)
        # pp(_speech_token_ids)
        # pp(_text_token_ids.shape)
        # pp(_llm_word_ids.shape)
        # pp(_speech_token_ids.shape)
        # pp(_text_label_ids.shape)
        # pp(_speech_label_ids.shape)
        # pp(_speech_label_ids_mask.shape)
        # assert False, "stop"
    # pad with ignore index for the labels of the whole batch
    bsz = len(text_input_ids)
    # print(text_speech_input_ids)
    padded_text_labels = pad_sequence(
        text_label_ids,
        batch_first=True,
        padding_value=ignore_idx,
    ) # labels should use padding value=-1
    padded_speech_labels = pad_sequence(
        speech_label_ids,
        batch_first=True,
        padding_value=ignore_idx,
    ) # labels should use padding value=-1
    padded_speech_labels_mask = pad_sequence(
        speech_label_ids_mask,
        batch_first=True,
        padding_value=False,
    )
    # print(padded_speech_labels_mask.shape)
    # print(padded_speech_labels.shape)
    # assert False, "stop"
    # print(padded_text_labels, padded_text_labels.shape)
    # print(padded_speech_labels, padded_speech_labels.shape)
    # generate attention masks of the padded input (by inspecting the ignore index)
    text_attention_mask_bool = (padded_text_labels != ignore_idx)
    speech_attention_mask_bool = (padded_speech_labels != ignore_idx)[:, :, 0]
    # print(text_attention_mask_bool)
    # print(speech_attention_mask_bool)
    # print((text_attention_mask_bool == speech_attention_mask_bool).sum())
    # clone the labels and generate input tokens by transforming ignore_idx to pad_idx
    padded_text_input_ids = pad_sequence(
        text_input_ids,
        batch_first=True,
        padding_value=text_pad_idx,
    )
    # padded_text_input_ids[~text_attention_mask_bool] = text_eos_idx
    padded_speech_input_ids = pad_sequence(
        speech_input_ids,
        batch_first=True,
        padding_value=speech_pad_idx,
    )
    # padded_speech_input_ids[~speech_attention_mask_bool] = speech_eos_idx
    # print(padded_text_input_ids)
    # print(padded_speech_input_ids)
    # change attention mask to proper dtype
    text_attention_mask = text_attention_mask_bool.long()
    speech_attention_mask = speech_attention_mask_bool.long()
    padded_inputs = {
        'text_input_ids': padded_text_input_ids,
        'text_attention_mask': text_attention_mask,
        'text_labels': padded_text_labels,
        'text_input_ids_lens': torch.tensor(text_input_ids_lens),
        'speech_input_ids': padded_speech_input_ids,
        'speech_attention_mask': speech_attention_mask,
        'speech_labels': padded_speech_labels,
        'speech_labels_mask': padded_speech_labels_mask,
        'speech_input_ids_lens': torch.tensor(speech_input_ids_lens),
    }
    if device != None:
        for key, val in padded_inputs.items():
            padded_inputs[key] = val.to(device)

    return padded_inputs


def pad_seq_collate_fn_for_taste_repeat(
    batch, 
    device=None,
    map_orig_eos_to_special_token_id=None, # map <|eos|> to <|reserved_special_token_i|> (the first one is 128002) to avoid using eos
    mask_second_text_labels=True,
    ensure_alignment=False,
    force_recompute_delayed=True,
    use_delayed_token=True,
    add_bos=True,
    add_eos=True,
    text_pad_idx=None,   # set to none to be the same as the eos idx
    text_bos_idx=128000,
    text_eos_idx=128001,
    speech_pad_idx=None, # set to none to be the same as the eos idx
    speech_bos_idx=4096,
    speech_eos_idx=4097,
    ignore_idx=-1,
    speech_token_column_name="llm_aligned_taste_token_ids",
    text_token_column_name="llm_text_token_ids"
):
    # for each single text, speech token pairs
    text_input_ids = []
    text_label_ids = []
    text_input_ids_lens = []
    speech_input_ids = []
    speech_label_ids = []
    speech_label_ids_mask = []
    speech_input_ids_lens = []
    if text_pad_idx == None:
        text_pad_idx = text_eos_idx
    if speech_pad_idx == None:
        speech_pad_idx = speech_eos_idx
    if map_orig_eos_to_special_token_id is not None:
        if isinstance(map_orig_eos_to_special_token_id, list):
            first_special_idx = map_orig_eos_to_special_token_id[0]
            second_special_idx = map_orig_eos_to_special_token_id[-1]
        else:
            first_special_idx = map_orig_eos_to_special_token_id
            second_special_idx = map_orig_eos_to_special_token_id
    for sample in batch:
        # handle text_input
        _raw_text_token_ids = torch.tensor(sample[text_token_column_name])
        _first_text_token_ids = _raw_text_token_ids.clone()
        ## if map_orig_eos_to_special_token_id is set and found eos token in input, map the eos token to the assigned special token
        _second_text_token_ids = _first_text_token_ids.clone()
        if map_orig_eos_to_special_token_id is not None:
            _map_mask = _first_text_token_ids == text_eos_idx
            _first_text_token_ids[_map_mask] = first_special_idx
            _map_mask = _second_text_token_ids == text_eos_idx
            _second_text_token_ids[_map_mask] = second_special_idx
        _text_token_ids = torch.cat([_first_text_token_ids, _second_text_token_ids])
        if add_bos:
            _text_token_ids = torch.cat([torch.tensor([text_bos_idx]), _text_token_ids])
        if add_eos:
            _text_token_ids = torch.cat([_text_token_ids, torch.tensor([text_eos_idx])])
        # handle text_label_ids
        _text_label_ids = _text_token_ids.clone()
        if mask_second_text_labels:
            _position = len(_raw_text_token_ids) + add_bos - 1
            _mask = torch.arange(len(_text_label_ids)) > _position
            _text_label_ids[_mask] = ignore_idx
        _text_token_ids_len = len(_text_token_ids)
        # handle text label
        # _text_label_ids[-1] = ignore_idx
        # handle speech_input -> word-level token, need to handle delayed-by-word
        ## the aligned ones are for labels. <- need more consideration
        _aligned_speech_token_ids = torch.tensor(sample[speech_token_column_name]) # the `aligned` ones. should be used to be the labels
        _tsz, _qsz = _aligned_speech_token_ids.shape
        ## if add bos or add eos, add bos and eos (both modality)
        ## now handle the input. should be the word-level delayed ones.
        if sample.get('llm_delayed_taste_token_ids', None) is None or force_recompute_delayed:
            _llm_word_ids = np.array(sample['llm_word_ids'])
            _word_id_to_taste_token_dict = {}
            _prev_llm_wrd_id = min(_llm_word_ids) - 1 # _asr_wrd_id might start from 1, but it's okay. we just want the first initial wrd is different from the prev one. 
            _word_id_to_taste_token_dict[_prev_llm_wrd_id] = np.ones(_qsz, dtype=np.int32) * speech_pad_idx
            for _llm_wrd_id, _taste_token in zip(_llm_word_ids, _aligned_speech_token_ids):
                if _llm_wrd_id == _prev_llm_wrd_id: continue # word_id=_asr_wrd_id already has taste token registered
                _word_id_to_taste_token_dict[_llm_wrd_id] = _taste_token
                _prev_llm_wrd_id = _llm_wrd_id
            _delayed_speech_token_ids = np.zeros((_tsz, _qsz), dtype=np.int32)
            for _cur_idx, _prev_llm_wrd_id in enumerate(_llm_word_ids - 1):
                _prev_taste_token = _word_id_to_taste_token_dict[_prev_llm_wrd_id]
                _delayed_speech_token_ids[_cur_idx] = _prev_taste_token
        else:
            _delayed_speech_token_ids = sample['llm_delayed_taste_token_ids']
        _delayed_speech_token_ids = torch.tensor(_delayed_speech_token_ids)

        _first_speech_token_ids = torch.ones(_tsz, _qsz, dtype=_aligned_speech_token_ids.dtype) * speech_pad_idx
        _second_speech_token_ids = _delayed_speech_token_ids
        _speech_token_ids = torch.cat([_first_speech_token_ids, _second_speech_token_ids])

        _pad_for_speech_tokens = torch.ones(1, _qsz, dtype=_speech_token_ids.dtype) * speech_pad_idx
        if add_bos:
            _speech_token_ids = torch.cat([_pad_for_speech_tokens, _speech_token_ids])
        if add_eos:
            _speech_token_ids = torch.cat([_speech_token_ids, _pad_for_speech_tokens]) # add eos should be set to allow shift by one
        _speech_label_ids = torch.cat([_first_speech_token_ids, _aligned_speech_token_ids])
        _speech_label_ids = torch.cat([_pad_for_speech_tokens, _speech_label_ids]) # shift by one first to avoid shift when calculating labels
        if add_bos:
            _speech_label_ids = torch.cat([_pad_for_speech_tokens, _speech_label_ids]) # shift for bos
        _speech_label_ids_mask = torch.zeros(_speech_token_ids.size(0), dtype=torch.bool)
        _llm_word_ends_mask = torch.tensor(sample['llm_word_ends_mask'], dtype=torch.bool)
        _speech_label_ids_mask[-_tsz:] = _llm_word_ends_mask
        _speech_token_ids_len = len(_speech_token_ids)


        # add the last token to the _delayed ones if use_delayed token as input and output
            # for _dl_id, _al_id, _llm_wrd_id in zip(_delayed_speech_token_ids[:-1], _aligned_speech_token_ids, sample['llm_word_ids']):
            #     print(_dl_id, _al_id, _llm_wrd_id)
            # assert False, "stop"
    
        ## if ensure alignment: pad the short one to the longer one (with pad idx)
        if ensure_alignment:
            assert _speech_token_ids_len == _text_token_ids_len
            ### pad speech to text length
        ## add to list
        text_input_ids.append(_text_token_ids)
        text_label_ids.append(_text_label_ids)
        text_input_ids_lens.append(_text_token_ids_len)
        speech_input_ids.append(_speech_token_ids)
        speech_label_ids.append(_speech_label_ids)
        speech_label_ids_mask.append(_speech_label_ids_mask)
        speech_input_ids_lens.append(_speech_token_ids_len)
        # for _taste_token, _text_token, _taste_token_lb, _text_token_lb, _taste_label_mask in zip(_speech_token_ids[:-1], _text_token_ids[:-1], _speech_label_ids[1:], _text_label_ids[1:], _speech_label_ids_mask[1:]):
        #     print(_taste_token, _text_token, _taste_token_lb, _text_token_lb, _taste_label_mask)
        # assert False, "stop for debug"
        # pp(_text_token_ids)
        # pp(_text_label_ids)
        # pp(_llm_word_ids)
        # pp(_speech_label_ids_mask)
        # pp(_speech_label_ids)
        # pp(_speech_token_ids)
        # pp(_text_token_ids.shape)
        # pp(_llm_word_ids.shape)
        # pp(_speech_token_ids.shape)
        # pp(_text_label_ids.shape)
        # pp(_speech_label_ids.shape)
        # pp(_speech_label_ids_mask.shape)
        # assert False, "stop"
    # pad with ignore index for the labels of the whole batch
    bsz = len(batch)
    # print(text_speech_input_ids)
    padded_text_labels = pad_sequence(
        text_label_ids,
        batch_first=True,
        padding_value=ignore_idx,
    ) # labels should use padding value=-1
    padded_speech_labels = pad_sequence(
        speech_label_ids,
        batch_first=True,
        padding_value=ignore_idx,
    ) # labels should use padding value=-1
    padded_speech_labels_mask = pad_sequence(
        speech_label_ids_mask,
        batch_first=True,
        padding_value=False,
    )
    # print(padded_speech_labels_mask.shape)
    # print(padded_speech_labels.shape)
    # assert False, "stop"
    # print(padded_text_labels, padded_text_labels.shape)
    # print(padded_speech_labels, padded_speech_labels.shape)
    # generate attention masks of the padded input (by inspecting the ignore index)
    text_attention_mask_bool = (padded_text_labels != ignore_idx)
    speech_attention_mask_bool = (padded_speech_labels != ignore_idx)[:, :, 0]
    # print(text_attention_mask_bool)
    # print(speech_attention_mask_bool)
    # print((text_attention_mask_bool == speech_attention_mask_bool).sum())
    # clone the labels and generate input tokens by transforming ignore_idx to pad_idx
    padded_text_input_ids = pad_sequence(
        text_input_ids,
        batch_first=True,
        padding_value=text_pad_idx,
    )
    # padded_text_input_ids[~text_attention_mask_bool] = text_eos_idx
    padded_speech_input_ids = pad_sequence(
        speech_input_ids,
        batch_first=True,
        padding_value=speech_pad_idx,
    )
    # padded_speech_input_ids[~speech_attention_mask_bool] = speech_eos_idx
    # print(padded_text_input_ids)
    # print(padded_speech_input_ids)
    # change attention mask to proper dtype
    text_attention_mask = text_attention_mask_bool.long()
    speech_attention_mask = speech_attention_mask_bool.long()
    padded_inputs = {
        'text_input_ids': padded_text_input_ids,
        'text_attention_mask': text_attention_mask,
        'text_labels': padded_text_labels,
        'text_input_ids_lens': torch.tensor(text_input_ids_lens),
        'speech_input_ids': padded_speech_input_ids,
        'speech_attention_mask': speech_attention_mask,
        'speech_labels': padded_speech_labels,
        'speech_labels_mask': padded_speech_labels_mask,
        'speech_input_ids_lens': torch.tensor(speech_input_ids_lens),
    }
    if device != None:
        for key, val in padded_inputs.items():
            padded_inputs[key] = val.to(device)

    return padded_inputs


def _find_all_linear_names(model, lora_linear_names_to_skip=[]):
    cls = (torch.nn.Linear, )
    _lora_module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            or "Linear" in module.__class__.__name__
            and module.__class__.__name__ not in ("LlamaLinearScalingRotaryEmbedding",)
        ):
            names = name.split(".")
            _lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    lora_module_names = []
    # if output_embedding in lora_module_names:  # needed for 16-bit
    #     lora_module_names.remove(output_embedding)
    for name in _lora_module_names:
        if name not in lora_linear_names_to_skip:
            lora_module_names.append(name)
        else:
            print(f"{name} is in lora_linear_names_to_skip. will skip and not adopt lora for it.")
    # return filtered lora_module_names to be adapted with LoRA
    return list(lora_module_names)


def get_lora_config(model, training_config, inference=False):
    lora_target_modules = list(training_config.lora_target_modules or [])

    if training_config.lora_target_linear:
        linear_names = _find_all_linear_names(model, lora_linear_names_to_skip=getattr(training_config, "lora_linear_names_to_skip", []))
        print(f"found linear modules: {repr(linear_names)}")
        lora_target_modules = list(set(lora_target_modules + linear_names))

    lora_config = LoraConfig(
        r=training_config.lora_r,
        lora_alpha=training_config.lora_alpha,
        target_modules=lora_target_modules,
        layers_to_transform=None,
        lora_dropout=training_config.lora_dropout,
        fan_in_fan_out=training_config.lora_fan_in_fan_out,
        modules_to_save=training_config.lora_modules_to_save if training_config.lora_modules_to_save else None,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return lora_config


def load_training_config(pretrained_dir):
    _training_config_fpath = glob.glob(os.path.join(pretrained_dir, "*.yaml"))[0]
    _training_config_dict = yaml.safe_load(open(_training_config_fpath, 'rb').read())
    training_config = Namespace()
    for key, val in _training_config_dict.items():
        setattr(training_config, key, val)
    return training_config


def get_s3_speech_tokenizer(speech_tokenizer_pretrained_dir, device='cpu'):
    if device is None:
        print("Warning: will not use gpu becuase device is not set. use device=cpu")
        device = 'cpu'
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    # self.campplus_session = onnxruntime.InferenceSession(campplus_model, sess_options=option, providers=["CPUExecutionProvider"])
    speech_tokenizer_model_fpath = os.path.join(speech_tokenizer_pretrained_dir, f"speech_tokenizer_v1.onnx")
    # load s3 tokenizer
    speech_tokenizer_session = onnxruntime.InferenceSession(speech_tokenizer_model_fpath, sess_options=option, providers=["CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"])
    tts_config_fpath = os.path.join(speech_tokenizer_pretrained_dir, f"cosyvoice.yaml")
    with open(tts_config_fpath, 'r') as fr:
        tts_config = load_hyperpyyaml(fr)
    flow = tts_config['flow']
    hift = tts_config['hift']
    flow.to(device)
    hift.to(device)
    _flow_state_dict = torch.load(os.path.join(speech_tokenizer_pretrained_dir, 'flow.pt'), map_location=device)
    flow.load_state_dict(_flow_state_dict)
    _hift_state_dict = torch.load(os.path.join(speech_tokenizer_pretrained_dir, 'hift.pt'), map_location=device)
    hift.load_state_dict(_hift_state_dict)
    flow.eval()
    hift.eval()
    return speech_tokenizer_session, flow, hift


def get_spk_emb_extractor(speech_tokenizer_pretrained_dir):
    option = onnxruntime.SessionOptions()
    option.inter_op_num_threads = 1
    option.intra_op_num_threads = 1
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL # TODO: Should I keep this?
    providers = ["CPUExecutionProvider"]
    spk_emb_onnx_fpath = os.path.join(speech_tokenizer_pretrained_dir, f"campplus.onnx")
    spk_emb_ort_session = onnxruntime.InferenceSession(spk_emb_onnx_fpath, sess_options=option, providers=providers)
    return spk_emb_ort_session


def get_taste_speech_tokenizer(speech_tokenizer_pretrained_dir, speech_decoder_pretrained_dir, device='cpu'):
    # config fpath of speech tokenizer
    if device is None:
        print("Warning: will not use gpu becuase device is not set. use device=cpu")
        device = 'cpu'
    speech_tokenizer_config_fpath = os.path.join(speech_tokenizer_pretrained_dir, "config.yaml")
    speech_tokenizer_ckpt = os.path.join(speech_tokenizer_pretrained_dir, 'checkpoint_best.pt')
    with open(speech_tokenizer_config_fpath, 'r') as fr:
        speech_tokenizer_configs = load_hyperpyyaml(fr)
    taste_llm = speech_tokenizer_configs['llm']
    flow = speech_tokenizer_configs['flow']
    hift = speech_tokenizer_configs['hift']
    taste_llm.to(device)
    flow.to(device)
    hift.to(device)
    _taste_llm_state_dict = torch.load(speech_tokenizer_ckpt, map_location=device)
    taste_llm.load_state_dict(_taste_llm_state_dict, load_partial_list=[]) # fully loaded
    _flow_state_dict = torch.load(os.path.join(speech_decoder_pretrained_dir, 'flow.pt'), map_location=device)
    flow.load_state_dict(_flow_state_dict)
    _hift_state_dict = torch.load(os.path.join(speech_decoder_pretrained_dir, 'hift.pt'), map_location=device)
    hift.load_state_dict(_hift_state_dict)
    taste_llm.eval()
    flow.eval()
    hift.eval()
    add_eos = False
    strip_text = False
    with open(speech_tokenizer_config_fpath, 'r') as fr:
        for l in fr:
            if ("add_eos" in l) and ("True" in l):
                add_eos = True
                print(f"`add_eos` = True in {speech_tokenizer_config_fpath}. Will also add eos when extracting taste tokens.")
            if ("strip_text" in l) and ("True" in l):
                strip_text = True
                print(f"`strip_text` = True in {speech_tokenizer_config_fpath}. Will also strip text when extracting taste tokens.")
    taste_tokenizer_kwargs = {
        'add_eos': add_eos,
        'strip_text': strip_text
    }
    return taste_llm, flow, hift, taste_tokenizer_kwargs


def _extract_token_by_words(words_with_space, tokenizer, add_eos=True):
    word_level_token_ids = tokenizer(words_with_space, add_special_tokens=False).input_ids
    token_ids = []
    word_ids = []
    if add_eos:
        word_level_token_ids.append([tokenizer.eos_token_id])
    for _wrd_idx, _wrd_ids in enumerate(word_level_token_ids):
        token_ids.extend(_wrd_ids)
        word_ids.extend([_wrd_idx] * len(_wrd_ids))
    return token_ids, word_ids


def _aligned_taste_from_asr_to_llm(asr_token_ids, asr_word_ids, llm_token_ids, llm_word_ids, raw_taste_token_ids):
    # currently does not support drop eos before llm
    assert len(raw_taste_token_ids) == len(asr_word_ids), f"lengths between taste_token_raw and asr_word_ids are mismatched. \ntaste_token_raw: {raw_taste_token_ids} \nasr_word_ids:{asr_word_ids}"
    assert min(asr_word_ids) == min(llm_word_ids) and max(asr_word_ids) == max(llm_word_ids), f"word_ids between llm and asr are different! please check the data)."
    qsz = raw_taste_token_ids.shape[-1]
    tsz = len(llm_word_ids)
    _device, _dtype = raw_taste_token_ids.device, raw_taste_token_ids.dtype
    llm_aligned_taste_token = torch.zeros((tsz, qsz), device=_device, dtype=_dtype)
    llm_delayed_taste_token = torch.zeros((tsz, qsz), device=_device, dtype=_dtype)
    ## prepare word_id --> taste_token mapping
    word_id_to_taste_token_dict = {}
    prev_asr_wrd_id = min(asr_word_ids) - 1 # _asr_wrd_id might start from 1, but it's okay. we just want the first initial wrd is different from the prev one. 
    word_id_to_taste_token_dict[prev_asr_wrd_id] = torch.zeros(qsz, device=_device, dtype=_dtype)
    for _asr_wrd_id, _taste_token in zip(asr_word_ids, raw_taste_token_ids):
        if _asr_wrd_id == prev_asr_wrd_id: continue # word_id=_asr_wrd_id already has taste token registered
        word_id_to_taste_token_dict[_asr_wrd_id] = _taste_token
        prev_asr_wrd_id = _asr_wrd_id
    # assign taste_token according to llm_word_ids
    prev_llm_wrd_id = min(llm_word_ids) - 1
    for _cur_idx, _llm_wrd_id in enumerate(llm_word_ids):
        _taste_token = word_id_to_taste_token_dict[_llm_wrd_id] # take out the taste token corresponding to the word id
        _prev_taste_token = word_id_to_taste_token_dict[_llm_wrd_id - 1]
        llm_aligned_taste_token[_cur_idx] = _taste_token
        llm_delayed_taste_token[_cur_idx] = _prev_taste_token
    llm_word_ids_npy = np.array(llm_word_ids)
    llm_word_starts_mask = np.zeros_like(llm_word_ids_npy, dtype=bool)
    llm_word_ends_mask = np.zeros_like(llm_word_ids_npy, dtype=bool)
    # create start mask
    llm_word_starts_mask[0] = True
    llm_word_starts_mask[1:] = llm_word_ids_npy[1:] != llm_word_ids_npy[:-1] # if the cur one is different from the prev one, then it is a start of a word.
    # create end mask
    llm_word_ends_mask[-1] = True
    llm_word_ends_mask[:-1] = llm_word_ids_npy[:-1] != llm_word_ids_npy[1:] # if the cur one is different from the next one, then it is an end of a word.
    return llm_aligned_taste_token, llm_delayed_taste_token, llm_word_starts_mask, llm_word_ends_mask



def get_taste_result( # get taste tokenization result (aligned with asr text tokens)
    speech_pt_16k, 
    pre_asr_result, 
    taste_tokenizer, 
    whisper_feature_extractor,
    whisper_tokenizer,
    taste_tokenizer_kwargs={},
    whisper_prefix_tokens=[50258, 50259, 50360, 50364],
    llm_tokenizer=None,
    device=None,
):
    speech_feats, speech_feats_len = whisper_feature_extractor(speech_pt_16k, [speech_pt_16k.shape[-1]])
    add_eos = taste_tokenizer_kwargs.get('add_eos', True)
    strip_text = taste_tokenizer_kwargs.get('strip_text', False)
    if strip_text:
        _words = pre_asr_result.strip().split(' ')
        words_with_space = [f" {wrd}" if i!=0 else wrd for i, wrd in enumerate(_words)]
    words_with_space = pre_asr_result['words']
    # extract asr tokens
    asr_token_ids, asr_word_ids = _extract_token_by_words(words_with_space, whisper_tokenizer, add_eos=add_eos)
    # prepare taste input
    _whisper_text_tokens = whisper_prefix_tokens + asr_token_ids
    whisper_text_token = torch.tensor([_whisper_text_tokens])
    whisper_text_token_len = torch.tensor([len(_whisper_text_tokens)], dtype=torch.int32)
    asr_word_ids_pt = torch.tensor([asr_word_ids])
    # start taste_tokenization
    # with torch.inference_mode():
    with torch.inference_mode(), torch.amp.autocast('cuda'):
        speech_feats = speech_feats.to(device)
        speech_feats_len = speech_feats_len.to(device)
        tokenized_results = taste_tokenizer(
            speech_feats, speech_feats_len,
            None, None, None,
            whisper_text_token=whisper_text_token.to(device),
            whisper_text_token_len=whisper_text_token_len.to(device),
            words_index=None,
            word_ids=asr_word_ids_pt.to(device)
        )
    quantized_indices = tokenized_results['quantized_results']['quantized_indices']
    quantized_feats_lengths = tokenized_results['quantized_results']['quantized_feats_lengths']
    assert quantized_indices.shape[0] == 1, f"currently only support single sample processing"
    raw_taste_token_ids = quantized_indices.squeeze(0)
    result = {
        'add_eos': add_eos,
        'asr_text_token_ids': np.array(asr_token_ids, dtype=np.int32),
        'asr_text_token_ids_len': len(asr_token_ids),
        'asr_word_ids': np.array(asr_word_ids, dtype=np.int32),
        'raw_taste_token_ids': raw_taste_token_ids.cpu().numpy(),
    }
    if llm_tokenizer is not None:
        # convert taste aligned with asr tokens to aligned with llm tokens
        ## extract llm tokens
        llm_token_ids, llm_word_ids = _extract_token_by_words(words_with_space, llm_tokenizer, add_eos=add_eos)
        llm_aligned_taste_token, llm_delayed_taste_token, llm_word_starts_mask, llm_word_ends_mask = _aligned_taste_from_asr_to_llm(
            asr_token_ids, 
            asr_word_ids,
            llm_token_ids,
            llm_word_ids,
            raw_taste_token_ids
        )
        result.update({
            'llm_text_token_ids': np.array(llm_token_ids, dtype=np.int32),
            'llm_text_token_ids_len': len(llm_token_ids),
            'llm_word_ids': np.array(llm_word_ids),
            'llm_word_starts_mask': llm_word_starts_mask,
            'llm_word_ends_mask': llm_word_ends_mask,
            'llm_aligned_taste_token_ids': llm_aligned_taste_token.cpu().numpy(),
            'llm_delayed_taste_token_ids': llm_delayed_taste_token.cpu().numpy(),
        })
        # pp(quantized_indices)
        # print(asr_word_ids, llm_word_ids, sep='\n')
        # print(asr_token_ids, llm_token_ids, sep='\n')
        # pp(llm_aligned_taste_token)
        # pp(llm_delayed_taste_token)
        # for _llm_tid, _llm_wid, _llm_alt, _llm_dlt in zip(llm_token_ids, llm_word_ids, llm_aligned_taste_token, llm_delayed_taste_token):
        #     print(_llm_tid, _llm_wid, _llm_alt, _llm_dlt, sep="\t")
        # assert False, "stop for debug"
    return result


# def prepare_lora_model(model, training_config):
#     lora_config = get_lora_config(model, training_config)
#     peft_model = get_peft_model(model, lora_config)
#     peft_model.unfreeze_modules(training_config.modules_to_finetune)
#     messages = [('[O] ' if params.requires_grad else '[X] ') + name for name, params in peft_model.base_model.named_parameters()]
#     messages_from_orig_model = [('[O] ' if params.requires_grad else '[X] ') + name for name, params in peft_model.base_model.model.named_parameters()]
#     with open(training_config.exp_dir + '/weight_grad.txt', 'w') as fw:
#         fw.write('\n'.join(messages))
#         fw.write('\n============================================\n')
#         fw.write('\n'.join(messages_from_orig_model))
#         fw.write(f"\n{peft_model}\n")
#         # fw.write(f"{peft_model.base_model.model.lm_head}")
#     peft_model.print_trainable_parameters()
#     return peft_model


def prepare_dataset(data_list_fpath):
    data_list = []
    with open(data_list_fpath, 'r') as fr:
        for l in fr:
            ds_fpath = l.split(' ')[0].strip()
            data_list.append(ds_fpath)
    datasets = [Dataset.from_file(ds_fpath) for ds_fpath in tqdm(data_list, total=len(data_list), desc=f"loading datasets...")]
    print("concatenating datasets...")
    concat_dataset = concatenate_datasets(datasets)
    print("datasets concatenated.")
    return concat_dataset


if __name__ == "__main__":
    from datasets import Dataset
    from torch.utils.data import DataLoader
    # arrow_fpath = "/proj/gpu_d_09023_MR_dataset_augmented/emilia/en/s3_token_for_baseline/parallel/00000-00012/emilia-dataset-train-00071-of-04908-taste-llm.arrow"
    arrow_fpath = "/proj/gpu_d_09023_MR_dataset_augmented/emilia/en/taste_token/0207_stg2_eos_rvq-d256-l4-k512_sum_smaller-lr/combined_with_delayed/emilia-dataset-train-00071-of-04908-taste-llm.arrow"
    ds = Dataset.from_file(arrow_fpath)
    partial_fn_for_collate = partial(
        pad_seq_collate_fn_for_taste_repeat,
        map_orig_eos_to_special_token_id=[128002, 128003],
        add_bos=True,
        add_eos=True,
        force_recompute_delayed=True,
        use_delayed_token=True,
        ensure_alignment=True,
        speech_pad_idx=0, # set to none to be the same as the eos idx
        speech_bos_idx=0,
        speech_eos_idx=0,
    )
    dl = DataLoader(
        ds, 
        batch_size=3,
        shuffle=False,
        collate_fn=partial_fn_for_collate
    )
    for batch in dl:
        print(batch)
        pp(batch['text_input_ids'][0])
        pp(batch['text_input_ids_lens'][0])
        pp(batch['text_attention_mask'][0])
        pp(batch['text_labels'][0])
        print("------------------------------------")
        pp(batch['speech_input_ids'][0])
        pp(batch['speech_input_ids_lens'][0])
        pp(batch['speech_attention_mask'][0])
        pp(batch['speech_labels'][0])
        break



