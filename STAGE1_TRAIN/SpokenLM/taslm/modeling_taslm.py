import os
import glob
import yaml
import torch
import whisper
import torchaudio
import onnxruntime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
from pprint import pp
from functools import partial
from peft import get_peft_model
from transformers import PreTrainedModel, LlamaForCausalLM, LlamaModel, LlamaPreTrainedModel, AutoTokenizer, pipeline
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from funasr.frontends.whisper_frontend import WhisperFrontend
from cosyvoice.audio.audio_quantizer import RVQAudioQuantizer
from cosyvoice.utils.file_utils import load_wav
from .utils_taslm import pad_seq_collate_fn, pad_seq_collate_fn_for_taste, pad_seq_collate_fn_for_taste_repeat, get_taste_speech_tokenizer, get_s3_speech_tokenizer, get_taste_result, get_spk_emb_extractor
from .configuration_taslm import TaslmConfig
from .modules_taslm import FUSION_METHOD_CLASS_MAP, LatentSamplingLayer


class TaslmForCausalLM(PreTrainedModel):
    config_class = TaslmConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values", "causal_mask"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _tied_weights_keys = ["language_model.lm_head.weight", "lm_head.base_layer.weight"] # orig: ["lm_head.weight"]. -> tune also the lm_head during peft

    def __init__(self, config: TaslmConfig):
        super().__init__(config)
        self.torch_dtype = eval(f"torch.{config.llama_config.torch_dtype}")
        if config.llama_use_liger_kernel:
            self.language_model = AutoLigerKernelForCausalLM.from_pretrained(
                config.llama_pretrained_dir,
                torch_dtype=self.torch_dtype,
                attn_implementation=config.attn_implementation
            )
        else:
            self.language_model = LlamaForCausalLM.from_pretrained(
                config.llama_pretrained_dir,
                torch_dtype=self.torch_dtype,
                attn_implementation=config.attn_implementation
            ) # model, lm_head
        self.speech_token_type = config.speech_token_type
        self.speech_num_channels = config.speech_num_channels
        self.speech_loss_apply_mask = config.speech_loss_apply_mask
        self.speech_labels_apply_quantization = config.speech_labels_apply_quantization
        self.speech_token_adopt_latent_sampling = False
        if self.speech_num_channels > 1:
            if config.speech_embed_directly_use_rvq:
                self.speech_embed_tokens = RVQAudioQuantizer(
                    **config.speech_tokenizer_rvq_kwargs
                )
                self.remove_rvq_proj_layers = config.speech_tokenizer_rvq_kwargs['dim'] == config.speech_tokenizer_rvq_kwargs['codebook_dim']
                _state_dict_fpath = os.path.join(config.speech_tokenizer_pretrained_dir, 'checkpoint_best.pt')
                _state_dict = torch.load(_state_dict_fpath, map_location='cpu')
                rvq_state_dict = {}
                for name, params in _state_dict.items():
                    if 'rvq' in name:
                        new_name = f"rvq" + name.split('rvq')[-1]
                        if self.remove_rvq_proj_layers and 'rvq.project' in new_name:
                            print(f"skip {new_name} because rvq.project* layers should be drop (dim==codebook_dim)")
                            continue
                        rvq_state_dict[new_name] = params
                pp(rvq_state_dict.keys())
                pp(self.speech_embed_tokens.state_dict().keys())
                self.speech_embed_tokens.load_state_dict(rvq_state_dict, strict=True)
                if self.speech_labels_apply_quantization:
                    # required  a linear for constructing hidden rvq feature
                    self.speech_rvq_hidden_proj_layer = nn.Linear(config.llama_config.hidden_size, config.speech_tokenizer_hidden_size)
                    self.loss_fct_for_rvq_recon = nn.MSELoss()
                self.freeze_speech_embed()
            else:
                self.speech_embed_tokens = nn.ModuleList([
                    nn.Embedding(config.speech_vocab_size, config.speech_tokenizer_hidden_size)
                ]),
            self.speech_token_embed_proj_layer = nn.Linear(config.speech_tokenizer_hidden_size, config.llama_config.hidden_size)
            self.speech_multi_channel_loss_decay_factor = config.speech_multi_channel_loss_decay_factor
            if config.speech_token_adopt_latent_sampling:
                self.speech_token_adopt_latent_sampling = True
                self.latent_dim = config.speech_tokenizer_rvq_kwargs['dim']
                speech_latent_sampler_kwargs = config.speech_latent_sampler_kwargs
                if speech_latent_sampler_kwargs is None:
                    speech_latent_sampler_kwargs = {}
                speech_latent_sampler_kwargs.update(
                    {
                        'lm_hidden_dim': config.hidden_size,
                        'latent_dim': self.latent_dim
                    }
                )
                self.speech_latent_sampler_kwargs = speech_latent_sampler_kwargs
                print(f"speech latent sampler kwargs: {speech_latent_sampler_kwargs}")
                self.speech_latent_sampler = LatentSamplingLayer(
                    **speech_latent_sampler_kwargs
                )
                print("will use latent sampling technique for next prediction.")
        else:
            self.speech_embed_tokens = nn.Embedding(config.speech_vocab_size, config.llama_config.hidden_size) # + 1 for padding
        # get fusion layer
        _fusion_cls = FUSION_METHOD_CLASS_MAP[config.fusion_method]
        self.fusion_layer = _fusion_cls(**config.fusion_kwargs)
        # currently only single head is supported
        if not self.speech_token_adopt_latent_sampling:
            self.speech_head = nn.Linear(config.llama_config.hidden_size, self.speech_num_channels * config.speech_vocab_size, bias=False) # use a single linear supports multi-head prediction
        self.text_conduct_kl_loss = config.text_conduct_kl_loss
        self.ignore_index = -1 # for ignoring padding during training NOTE: eos_token should not be ignored
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.post_init()
    
    def get_speech_embed(self):
        return self.speech_embed_tokens

    def freeze_speech_embed(self):
        for name, param in self.speech_embed_tokens.named_parameters():
            param.requires_grad = False
            print(f"[{name}] is frozen")
    
    def unfreeze_modules(self, modules_to_finetune):
        for module_name in modules_to_finetune:
            if module_name == "speech_embed_tokens":
                for name, param in self.speech_embed_tokens.named_parameters():
                    param.requires_grad = True
            elif module_name == "speech_head":
                for name, param in self.speech_head.named_parameters():
                    param.requires_grad = True
            else:
                raise NotImplementedError(f"{module_name} can not be unfreezed. ")
    
    def apply_lora(self, lora_config, training_config):
        peft_lm_model = get_peft_model(self.language_model, lora_config)
        peft_lm_model.print_trainable_parameters()
        self.language_model = peft_lm_model
        messages = [('[O] ' if params.requires_grad else '[X] ') + name for name, params in self.named_parameters()]
        # messages_from_orig_model = [('[O] ' if params.requires_grad else '[X] ') + name for name, params in self.named_parameters()]
        if hasattr(training_config, 'exp_dir'):
            with open(training_config.exp_dir + '/weight_grad.txt', 'w') as fw:
                fw.write('\n'.join(messages))
                fw.write('\n============================================\n')
                # fw.write('\n'.join(messages_from_orig_model))
                # fw.write('\n============================================\n')
                fw.write(f"\n{self}\n")
                # fw.write(f"{peft_model.base_model.model.lm_head}")
    

    def register_speech_tokenizer_decoder_and_asr(
        self, 
        pretrained_dir,
        training_config,
        speech_tokenizer_pretrained_dir=None,
        speech_decoder_pretrained_dir=None,
        asr_pretrained_dir="/proj/mtklmadm/models/whisper-large-v3",
        llm_pretrained_dir="/proj/mtklmadm/models/mtk53678/Llama-3.2-1B",
        device=None,
    ):
        if speech_tokenizer_pretrained_dir is None:
            speech_tokenizer_pretrained_dir = self.config.speech_tokenizer_pretrained_dir
        print(f"speaker embedding extractor registered (from {speech_tokenizer_pretrained_dir}).")
        if 's3' in self.speech_token_type:
            self.spk_emb_extractor = get_spk_emb_extractor(speech_tokenizer_pretrained_dir)
            self.speech_tokenizer_session, self.flow, self.hift = get_s3_speech_tokenizer(speech_tokenizer_pretrained_dir, device=device)
            # TODO: add register s3 speech decoder
            _collate_fn = pad_seq_collate_fn
        elif 'taste' in self.speech_token_type:
            self.spk_emb_extractor = get_spk_emb_extractor(speech_decoder_pretrained_dir)
            self.taste_llm, self.flow, self.hift, self.taste_tokenizer_kwargs = get_taste_speech_tokenizer(speech_tokenizer_pretrained_dir, speech_decoder_pretrained_dir, device=device)
            self.taste_tokenizer = self.taste_llm.audio_tokenizer
            # NOTE: self.taste_tokenizer is now in eval mode and is on `device`
            # prepare the whisper feature extractor as well
            self.whisper_feature_extractor = WhisperFrontend(
                whisper_model='large-v3',
                do_pad_trim=True,
                permute=True,
            )
            _collate_fn = eval(training_config.collate_fn_name)
        else:
            raise NotImplementedError(f"speech_token_type: {self.speech_token_type} is not supported.")
        _collate_fn_kwargs = training_config.collate_fn_kwargs # for controling how the speech tokens and text tokens are combined as input. please refer to the pretrained dir to find out.
        print(f"collate_fn_kwargs: {_collate_fn_kwargs}")
        self.collate_fn = partial(
            _collate_fn,
            **_collate_fn_kwargs,
        )
        self.collate_fn_kwargs = _collate_fn_kwargs
        print(f"speech tokenizer and collate function registered")
        self.asr_pipe = pipeline(
            'automatic-speech-recognition',
            model=asr_pretrained_dir,
            torch_dtype=torch.float16,
            # attn_implementation='flash_attention_2',
            device=device,
            chunk_length_s=30,
        )
        print(f"asr pipe registered (from {asr_pretrained_dir}).")
        self.asr_tokenizer = AutoTokenizer.from_pretrained(asr_pretrained_dir)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_pretrained_dir)
        print(f"text tokenizers registered (from {asr_pretrained_dir} and {llm_pretrained_dir})")
        self.asr_forced_decoder_ids = self.asr_tokenizer.get_decoder_prompt_ids(
            task='transcribe',
            language='en',
            no_timestamps=False,
        )
        print(f"asr_forced_decoder_ids={self.asr_forced_decoder_ids}")
    

    def register_llm_word_start_tokens(self):
        assert hasattr(self, 'llm_tokenizer'), "llm tokenizer is not yet registered."
        self.is_word_start_dict = {}
        for i in range(self.config.text_vocab_size):
            if i >= 128000: 
                self.is_word_start_dict[i] = True
                continue
            _subword = self.llm_tokenizer.decode(i)
            if _subword[0] == ' ':
                self.is_word_start_dict[i] = True
            else:
                self.is_word_start_dict[i] = False

    
    def _prepare_input(self, speech_pt_16k, pre_asr_result=None, device=None):
        if 's3' in self.speech_token_type:
            feat = whisper.log_mel_spectrogram(speech_pt_16k, n_mels=128)
            print(feat.shape)
            speech_token = self.speech_tokenizer_session.run(
                None, 
                {
                    self.speech_tokenizer_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                    self.speech_tokenizer_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)
                }
            )[0].flatten().tolist()
            _text = pre_asr_result['text'].strip()
            llm_text_token_ids = self.llm_tokenizer(_text, add_special_tokens=False).input_ids
            speech_token_column_name = self.collate_fn_kwargs['speech_token_column_name']
            text_token_column_name = self.collate_fn_kwargs['text_token_column_name']
            sample = {
                speech_token_column_name: speech_token,
                text_token_column_name: llm_text_token_ids,
            }
            batch_input = self.collate_fn(
                [sample],
                device=device
            )
        elif 'taste' in self.speech_token_type:
            taste_result = get_taste_result( # get taste tokenization result (aligned with asr text tokens)
                speech_pt_16k, 
                pre_asr_result, 
                self.taste_tokenizer, 
                self.whisper_feature_extractor,
                self.asr_tokenizer,
                device=device,
                llm_tokenizer=self.llm_tokenizer
            )
            batch_input = self.collate_fn(
                [taste_result],
                device=device
            )
            # pp(batch_input)
            # for _tid, _tlb, _sid, _slb, _smsk, _wid in zip(
            #     batch_input['text_input_ids'][0, :-1],
            #     batch_input['text_labels'][0, 1:],
            #     batch_input['speech_input_ids'][0, :-1, :],
            #     batch_input['speech_labels'][0, 1:, :],
            #     batch_input['speech_labels_mask'][0, 1:],
            #     list(taste_result['llm_word_ids']) + [-1],
            # ):
            #     print(_tid, _tlb, _sid, _slb, _smsk, _wid, sep='\t')
            # print(batch_input['speech_input_ids'].shape)
            # print(batch_input['speech_labels'].shape)
            # print(batch_input['speech_labels_mask'].shape)
            # assert False, "stop"
        else:
            raise NotImplementedError
        
        return batch_input
    

    def _pre_asr(self, speech_16k, cutoff_word_idx=None):
        result = self.asr_pipe(
            speech_16k, 
            generate_kwargs={
                'forced_decoder_ids': self.asr_forced_decoder_ids
            },
            return_timestamps='word',
        )[0]
        asr_text = result['text']
        asr_words = [chunk['text'] for chunk in result['chunks']]
        cutoff_speech_npy_16k = speech_16k[0]
        if cutoff_word_idx is not None:
            asr_words = []
            last_word_end_timestamp = 0.0
            for word_idx, chunk in enumerate(result['chunks']):
                if word_idx < cutoff_word_idx:
                    asr_words.append(chunk['text'])
                    last_word_end_timestamp = chunk['timestamp'][-1]
                else:
                    break
            asr_text = ''.join(asr_words)
            last_word_end_frame_idx = min(int(last_word_end_timestamp * 16_000), speech_16k[0].shape[-1])
            cutoff_speech_npy_16k = speech_16k[0][:last_word_end_frame_idx]
        return {
            'text': asr_text,
            'words': asr_words,
            'cutoff_speech_npy_16k': cutoff_speech_npy_16k
        }

    
    def _extract_spk_emb(self, speech_pt_16k):
        feat = kaldi.fbank(
            speech_pt_16k,
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.spk_emb_extractor.run(None, {self.spk_emb_extractor.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        return torch.tensor(embedding)


    def calculate_log_likelihood(self, speech_fpath, pre_asr_text=None, device=None):
        self.eval()
        speech_pt_16k = load_wav(speech_fpath, 16_000)
        if pre_asr_text is None:
            pre_asr_result = self._pre_asr([speech_pt_16k.squeeze(0).numpy()])
        else:
            _pre_asr_words = pre_asr_text.strip().split(' ')
            _pre_asr_words_with_spaces = [f" {wrd}" for wrd in _pre_asr_words]
            pre_asr_result = {
                'text': pre_asr_text,
                'words': _pre_asr_words_with_spaces,
            }
        batch_input = self._prepare_input(speech_pt_16k, pre_asr_result, device=device)
        with torch.inference_mode(), torch.amp.autocast(device_type=device):
            result = self.forward(**batch_input)
        
        log_likelihood_result = {}
        # calculate text log likelihood
        _text_label_ids = batch_input['text_labels'][..., 1:]
        _text_logits = result['text_logits'][..., :-1, :]
        _text_label_valid_mask = _text_label_ids != self.ignore_index
        text_label_ids = _text_label_ids[_text_label_valid_mask]
        reversed_text_label_ids = torch.flip(text_label_ids, [0])
        text_logits = _text_logits[_text_label_valid_mask]
        assert text_label_ids.shape[0] == text_logits.shape[0]
        text_log_likelihood = -(self.loss_fct(text_logits, text_label_ids)).item()
        reversed_text_log_likelihood = -(self.loss_fct(text_logits, reversed_text_label_ids)).item()
        log_likelihood_result['text_log_likelihood'] = text_log_likelihood
        log_likelihood_result['reversed_text_log_likelihood'] = reversed_text_log_likelihood
        # calculate speech log likelihood
        _speech_logits = result['speech_logits'][..., :-1, :]
        _speech_label_ids = batch_input['speech_labels'][:, 1:, ...] # the last can be one or multiple
        _speech_labels_mask = batch_input.get('speech_labels_mask', None)
        if _speech_labels_mask is not None:
            _speech_labels_mask = _speech_labels_mask[..., 1:]
        for i in range(self.speech_num_channels):
            if self.speech_num_channels > 1:
                _cur_speech_label_ids = _speech_label_ids[..., i]
            else:
                _cur_speech_label_ids = _speech_label_ids
            _channel_start_idx, _channel_end_idx = self.config.speech_vocab_size * i, self.config.speech_vocab_size * (i+1)
            _cur_speech_logits = _speech_logits[..., :, _channel_start_idx:_channel_end_idx]
            speech_label_valid_mask = _cur_speech_label_ids != self.ignore_index
            if _speech_labels_mask is not None:
                speech_label_valid_mask = torch.logical_and(speech_label_valid_mask, _speech_labels_mask)
            cur_speech_label_ids = _cur_speech_label_ids[speech_label_valid_mask]
            reversed_cur_speech_label_ids = torch.flip(cur_speech_label_ids, [0])
            cur_speech_logits = _cur_speech_logits[speech_label_valid_mask]
            assert cur_speech_label_ids.shape[0] == cur_speech_logits.shape[0]
            cur_speech_log_likelihood = -(self.loss_fct(cur_speech_logits, cur_speech_label_ids)).item()
            reversed_cur_speech_log_likelihood = -(self.loss_fct(cur_speech_logits, reversed_cur_speech_label_ids)).item()
            log_likelihood_result[f'speech_log_likelihood.{i}'] = cur_speech_log_likelihood
            log_likelihood_result[f'reversed_speech_log_likelihood.{i}'] = reversed_cur_speech_log_likelihood
        return log_likelihood_result
    

    def _prepare_input_for_ttslm(self, taslm_generation_result, spk_emb=None):
        if 'taste' in self.speech_token_type:
            # get delayed to aligned taste token with asr tokenizer
            assert taslm_generation_result['generated_text_ids'].shape[0] == 1, f"currently does not support batch decoding."
            generated_text_token_ids_for_decode   = taslm_generation_result['generated_text_ids'][0, 1:] # discard bos token
            llm_delayed_taste_token_for_decode = taslm_generation_result['generated_speech_ids'][0, 1:] # discard bos speech token
            _dtype = llm_delayed_taste_token_for_decode.dtype
            _device = llm_delayed_taste_token_for_decode.device
            # build llm_word_ids for alignedgeneration
            llm_word_ids = []
            llm_word_level_token_ids = []
            cur_llm_word_idx = 0 # word idx will starts from 1 typically. # NOTE: should add examination?
            cur_llm_word_ids_bucket = []
            for _text_id_pt in generated_text_token_ids_for_decode:
                _text_id = _text_id_pt.item()
                _is_wrd_start = self.is_word_start_dict[_text_id]
                if _is_wrd_start:
                    cur_llm_word_idx += 1
                    if len(cur_llm_word_ids_bucket) > 0:
                        llm_word_level_token_ids.append(cur_llm_word_ids_bucket)
                    cur_llm_word_ids_bucket = [_text_id]
                else:
                    cur_llm_word_ids_bucket.append(_text_id)
                llm_word_ids.append(cur_llm_word_idx)
            if len(cur_llm_word_ids_bucket) > 0:
                llm_word_level_token_ids.append(cur_llm_word_ids_bucket)
            # for _word_id, _text_id_pt, _delayed_taste_ids in zip(llm_word_ids, generated_text_token_ids_for_decode, llm_delayed_taste_token_for_decode):
            #     _text_id = _text_id_pt.item()
            #     print(_delayed_taste_ids, _word_id, _text_id, self.llm_tokenizer.decode(_text_id).replace(' ', '_'), sep='\t')
            # build up asr_word_level_token
            asr_word_ids = []
            word_ids_offset = min(llm_word_ids)
            asr_word_level_token_ids = []
            asr_text_token_ids = []
            _map_orig_eos_to_special_token_id = self.collate_fn_kwargs.get('map_orig_eos_to_special_token_id', self.config.llama_config.eos_token_id)
            for _idx, _llm_wrd_ids in enumerate(llm_word_level_token_ids):
                if _map_orig_eos_to_special_token_id in _llm_wrd_ids:
                    # hit the eos will break the loop after adding eos token
                    _new_asr_word_level_token_ids = [self.asr_tokenizer.eos_token_id]
                    asr_word_level_token_ids.append(_new_asr_word_level_token_ids)
                    asr_text_token_ids.extend(_new_asr_word_level_token_ids)
                    asr_word_ids.extend([_idx + word_ids_offset] * 1)
                    break
                _word_with_space = self.llm_tokenizer.decode(_llm_wrd_ids)
                _new_asr_word_level_token_ids = self.asr_tokenizer(_word_with_space, add_special_tokens=False).input_ids
                asr_word_level_token_ids.append(_new_asr_word_level_token_ids)
                asr_text_token_ids.extend(_new_asr_word_level_token_ids)
                asr_word_ids.extend([_idx + word_ids_offset] * len(_new_asr_word_level_token_ids))
            last_llm_wrd_idx_for_tts = _idx                
            # log for debug
            # for _llm_wrd_ids, _asr_wrd_ids in zip(llm_word_level_token_ids[:last_llm_wrd_idx_for_tts+1], asr_word_level_token_ids):
            #     _llm_word = self.llm_tokenizer.decode(_llm_wrd_ids)
            #     _asr_word = self.asr_tokenizer.decode(_asr_wrd_ids)
            #     print(_llm_word, _asr_word, sep='\t')
            assert min(asr_word_ids) == min(llm_word_ids) and max(asr_word_ids) == max(llm_word_ids)-1, f"word ids are not suitable for mapping delayed taste token to aligned taste token. please check the result.\n asr_word_ids: {asr_word_ids} \n llm_word_ids: {llm_word_ids}"
            assert len(llm_word_ids) == len(llm_delayed_taste_token_for_decode), f"llm_word_ids doesnot aligned with the generated_delayed_taste_token_length.\n llm_word_ids: {llm_word_ids}\n {llm_delayed_taste_token_for_decode}"
            # start building word_id to delayed_taste_token dict
            word_id_to_delayed_taste_token_dict = {}
            prev_llm_wrd_id = min(llm_word_ids) - 1
            for _llm_wrd_id, _delayed_taste_token in zip(llm_word_ids, llm_delayed_taste_token_for_decode):
                if _llm_wrd_id == prev_llm_wrd_id: continue
                word_id_to_delayed_taste_token_dict[_llm_wrd_id] = _delayed_taste_token
            _qsz = llm_delayed_taste_token_for_decode.shape[-1]
            _tsz = len(asr_word_ids)
            asr_aligned_taste_token_for_decode = torch.zeros((_tsz, _qsz), dtype=_dtype, device=_device)
            for _cur_idx, _asr_wrd_id in enumerate(asr_word_ids):
                asr_aligned_taste_token_for_decode[_cur_idx] = word_id_to_delayed_taste_token_dict[_asr_wrd_id + 1] # aligned the word-level taste token back!
            # log for debug
            # for _asr_wrd_id, _asr_aligned_taste_token in zip(asr_word_ids, asr_aligned_taste_token_for_decode):
            #     print(_asr_wrd_id, _asr_aligned_taste_token, sep='\t')
            # for _llm_wrd_id, _llm_delayed_taste_token in zip(llm_word_ids, llm_delayed_taste_token_for_decode):
            #     print(_llm_wrd_id, _llm_delayed_taste_token, sep='\t')
            return {
                'text_token': torch.tensor(asr_text_token_ids, dtype=_dtype, device=_device).view(1, -1),
                'text_token_len': torch.tensor([len(asr_text_token_ids)], dtype=_dtype, device=_device),
                'taste_token': asr_aligned_taste_token_for_decode.unsqueeze(0),
                'taste_token_len': torch.tensor([_tsz], device=_device),
                'embedding': torch.zeros((1, 192), device=_device) if spk_emb is None else spk_emb.to(_device).view(1, -1),
            }
        elif 's3' in self.speech_token_type:
            _device = taslm_generation_result['generated_speech_ids'].device
            return {
                'tts_speech_token': taslm_generation_result['generated_speech_ids'][:, 1:-1], # skip start and end
                'embedding': torch.zeros((1, 192), device=_device) if spk_emb is None else spk_emb.to(_device).view(1, -1),
            }
        else:
            raise NotImplementedError(f"{self.speech_token_type} is not implemented")

    
    def generate_speech(self, taslm_generation_result, spk_emb=None):
        input_for_ttslm = self._prepare_input_for_ttslm(taslm_generation_result, spk_emb=spk_emb)
        # for key, val in input_for_ttslm.items():
        #     print(f"{key}: {val.shape}")
        _device = input_for_ttslm['embedding'].device
        if 'taste' in self.speech_token_type:
            tts_speech_token = self.taste_llm.inference(
                text=input_for_ttslm['text_token'],
                text_len=input_for_ttslm['text_token_len'],
                prompt_text=torch.zeros(1, 0, dtype=torch.int32, device=_device), 
                prompt_text_len=torch.zeros(1, dtype=torch.int32, device=_device),
                prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32, device=_device), 
                prompt_speech_token_len=torch.zeros(1, dtype=torch.int32, device=_device),
                embedding=input_for_ttslm['embedding'],
                taste_token=input_for_ttslm['taste_token'],
                taste_token_len=input_for_ttslm['taste_token_len'],
            )
        elif 's3' in self.speech_token_type:
            tts_speech_token = input_for_ttslm['tts_speech_token']
        else:
            raise NotImplementedError(f"{self.speech_token_type} is not implemented")
        print(tts_speech_token)
        print(tts_speech_token.shape)
        tts_mel = self.flow.inference(
            token=tts_speech_token,
            token_len=torch.tensor([tts_speech_token.size(1)], dtype=torch.int32, device=_device),
            prompt_token=torch.zeros(1, 0, dtype=torch.int32, device=_device),
            prompt_token_len=torch.zeros(1, dtype=torch.int32, device=_device),
            prompt_feat=torch.zeros(1, 0, 80, device=_device),
            prompt_feat_len=torch.zeros(1, dtype=torch.int32, device=_device),
            embedding=input_for_ttslm['embedding'],
        )
        print(tts_mel)
        print(tts_mel.shape)
        tts_speech = self.hift.inference(mel=tts_mel).cpu()
        print(tts_speech.shape)
        return tts_speech
        

    # @torch.cuda.amp.autocast()
    # @torch.amp.autocast('cuda', enabled=False)
    def forward(
        self, 
        # Inputs for text
        text_input_ids=None,
        text_attention_mask=None,
        text_labels=None,
        # Inputs for speech
        speech_input_ids=None,
        speech_attention_mask=None,
        speech_labels=None,
        # other kwargs
        past_key_values=None,
        use_cache=None,
        **kwargs,
    ):
        # compute embeddings for each modality
        if text_input_ids is not None:
            embed_tokens = self.language_model.get_input_embeddings()
            text_embeds = embed_tokens(text_input_ids)
        else:
            text_embeds = 0 
        
        if speech_input_ids is not None:
            if self.speech_num_channels > 1:
                if isinstance(self.speech_embed_tokens, RVQAudioQuantizer):
                    _rvq_encoded_result = self.speech_embed_tokens.encode(speech_input_ids, None)
                    _encoded_speech = _rvq_encoded_result['quantized_feats']
                else:
                    raise NotImplementedError
                speech_embeds = self.speech_token_embed_proj_layer(_encoded_speech)
            else:
                speech_embeds = self.speech_embed_tokens(speech_input_ids)
        else:
            speech_embeds = 0
        
        combined_embeds = self.fusion_layer(text_embeds, speech_embeds)
        # combined_embeds = text_embeds + speech_embeds
        # pp("combined embeds")
        # pp(combined_embeds.shape)
        if text_attention_mask is not None and speech_attention_mask is not None:
            # Elementwise OR
            combined_attention_mask = ((text_attention_mask + speech_attention_mask) > 0).long() # make sure that the attention map is the OR of the both modalities
        elif text_attention_mask is not None:
            combined_attention_mask = text_attention_mask
        elif speech_attention_mask is not None:
            combined_attention_mask = speech_attention_mask
        else:
            combined_attention_mask = None
        # pp("combined attn mask")
        # pp(combined_attention_mask)
        # Forward pass through the Transformer
        # for peft 
        decoder = self.language_model.get_decoder()
        transformer_outputs = decoder(
            inputs_embeds = combined_embeds,
            attention_mask = combined_attention_mask,
            past_key_values = past_key_values,
            use_cache = use_cache, 
            # **kwargs,
        )
        hidden_states = transformer_outputs[0] # shape: (bsz, tsz, hidden_size)
        # pp("hidden_states")
        # pp(hidden_states.shape)
        # compute prediction logits
        text_logits = self.language_model.lm_head(hidden_states)
        speech_logits, speech_y_pred = None, None
        if self.speech_token_adopt_latent_sampling:
            mu, logvar, speech_y_pred = self.speech_latent_sampler(hidden_states)
        else:
            speech_logits = self.speech_head(hidden_states)

        # Compute losses
        loss = None
        loss_fct = self.loss_fct
        text_loss = 0.
        speech_loss = 0.
        loss_dict = {}

        if text_labels is not None:
            # For causal LM loss, shift logits and labels (token < n predicts token n)
            shift_text_logits = text_logits[..., :-1, :].contiguous()
            shift_text_labels = text_labels[..., 1:].contiguous()
            text_ce_loss = loss_fct(
                shift_text_logits.view(-1, self.config.text_vocab_size),
                shift_text_labels.view(-1)
            )
            loss_dict['text_ce_loss'] = text_ce_loss.item()
            text_loss += text_ce_loss
            # calculate text kl loss if set
            if self.text_conduct_kl_loss:
                try:
                    # with torch.inference_mode():
                    self.language_model.base_model.disable_adapter_layers()
                    _decoder = self.language_model.base_model.get_decoder()
                    orig_text_only_hidden_states = _decoder(
                        inputs_embeds = text_embeds,
                        attention_mask = combined_attention_mask,
                        past_key_values = past_key_values,
                        use_cache = use_cache, 
                        # **kwargs,
                    )[0] # take out the hidden states
                    orig_text_only_logits = self.language_model.base_model.lm_head(orig_text_only_hidden_states)
                    shift_orig_text_only_log_prob = F.log_softmax(orig_text_only_logits[..., :-1, :80000].contiguous(), dim=-1).detach()
                    self.language_model.base_model.enable_adapter_layers()
                    shift_text_log_prob = F.log_softmax(shift_text_logits[..., :80000], dim=-1)
                    shift_text_kl_loss_mask = (shift_text_labels == self.ignore_index) # index to ignore
                    src_text_log_prob = shift_text_log_prob[shift_text_kl_loss_mask]
                    tgt_text_log_prob = shift_orig_text_only_log_prob[shift_text_kl_loss_mask]
                    # print(src_text_log_prob[0, :10], src_text_log_prob.shape)
                    # print(tgt_text_log_prob[0, :10], tgt_text_log_prob.shape)
                    text_kl_loss = F.kl_div(src_text_log_prob, tgt_text_log_prob, log_target=True, reduction='batchmean') # exclude the special ones
                except Exception as e:
                    print(f"orig_text_logits shape: {orig_text_only_logits.shape}")
                    text_inputs_len = kwargs['text_input_ids_lens']
                    print(f"text input ids len: {text_inputs_len}")
                    raise e
                # print(text_kl_loss)
                # assert False, "stop for debug"
                loss_dict['text_kl_loss'] = text_kl_loss.item()
                text_loss += text_kl_loss
        
        if speech_labels is not None:
            if self.speech_num_channels > 1:
                _speech_labels_mask = kwargs.get('speech_labels_mask', None)
                if _speech_labels_mask is not None:
                    shift_speech_labels_mask_for_no_loss = ~(_speech_labels_mask[...,1:])
                if self.speech_token_adopt_latent_sampling:
                    # calculate L_reg
                    shift_speech_y_target = _encoded_speech[..., 1:, :]
                    shift_speech_y_pred = speech_y_pred[..., :-1, :]
                    shift_mu = mu[..., :-1, :]
                    shift_logvar = mu[..., :-1, :]
                    if _speech_labels_mask is not None and self.speech_loss_apply_mask:
                        shift_speech_y_target = shift_speech_y_target[~shift_speech_labels_mask_for_no_loss]
                        shift_speech_y_pred = shift_speech_y_pred[~shift_speech_labels_mask_for_no_loss]
                        shift_mu = shift_mu[~shift_speech_labels_mask_for_no_loss]
                        shift_logvar = shift_logvar[~shift_speech_labels_mask_for_no_loss]
                    l_reg = F.mse_loss(shift_speech_y_pred, shift_speech_y_target, reduction='mean')
                    # l_reg = l_reg / (shift_speech_y_pred.numel() / shift_speech_y_pred.shape[-1])
                    # calculate L_kl 
                    # 0.5 * sum(sigma^2 + (mu - y_target)^2 - 1 - log(sigma^2))
                    shift_sigma_sq = torch.exp(shift_logvar)
                    l_kl = 0.5 * torch.mean(
                        # torch.sum(shift_sigma_sq + (shift_mu - shift_speech_y_target)**2 - 1 - shift_logvar, dim=-1)
                        torch.mean(shift_sigma_sq + (shift_mu - shift_speech_y_target)**2 - 1 - shift_logvar, dim=-1) # we use mean to avoid too diverged value the loss term
                    )
                    speech_loss = l_reg + 2 * l_kl
                    loss_dict['speech_latent_reg_loss'] = l_reg.item()
                    loss_dict['speech_latent_kl_loss'] = l_kl.item()
                else:
                    _decay_factor = self.speech_multi_channel_loss_decay_factor
                    for i in range(self.speech_num_channels):
                        _channel_start_idx, _channel_end_idx = self.config.speech_vocab_size * i, self.config.speech_vocab_size * (i+1)
                        shift_speech_logits = speech_logits[..., :-1, _channel_start_idx:_channel_end_idx].contiguous()
                        shift_speech_labels = speech_labels[..., 1:, i].contiguous()
                        if _speech_labels_mask is not None and self.speech_loss_apply_mask:
                            # pp(shift_speech_labels[1])
                            shift_speech_labels[shift_speech_labels_mask_for_no_loss] = -1
                            # pp(shift_speech_labels[1])
                            # for _speech_id, _text_id, _speech_label, _text_label in zip(speech_input_ids[1, :-1, i], text_input_ids[1, :-1], shift_speech_labels[1], shift_text_labels[1]):
                            #     print(_speech_id, _text_id, _speech_label, _text_label)
                            # assert False, "stop for debug"
                        cur_speech_loss = _decay_factor**(i+1) * loss_fct(
                            shift_speech_logits.view(-1, self.config.speech_vocab_size),
                            shift_speech_labels.view(-1)
                        )
                        speech_loss += cur_speech_loss
                        loss_dict[f'speech_layer_{i}_ce_loss'] = cur_speech_loss.item()
                if self.speech_labels_apply_quantization:
                    # _encoded_speech_labels = self.speech_embed_tokens(speech_labels, None, apply_mask=True)['quantized_feats']
                    shift_encoded_speech_labels = _encoded_speech[..., 1:, :]
                    encoded_speech_recon = self.speech_rvq_hidden_proj_layer(hidden_states)[..., :-1, :]
                    if _speech_labels_mask is not None and self.speech_loss_apply_mask:
                        shift_encoded_speech_labels = shift_encoded_speech_labels[~shift_speech_labels_mask_for_no_loss]
                        encoded_speech_recon = encoded_speech_recon[~shift_speech_labels_mask_for_no_loss]
                    encoded_recon_loss = self.loss_fct_for_rvq_recon(
                        encoded_speech_recon, shift_encoded_speech_labels
                    )
                    loss_dict['speech_recon_loss'] = encoded_recon_loss.item()
                    speech_loss += _decay_factor * encoded_recon_loss
                # if hasattr(self, "")
            else:
                shift_speech_logits = speech_logits[..., :-1, :].contiguous()
                shift_speech_labels = speech_labels[..., 1:].contiguous()
                speech_loss += loss_fct(
                    shift_speech_logits.view(-1, self.config.speech_vocab_size),
                    shift_speech_labels.view(-1)
                )
                loss_dict['speech_ce_loss'] = speech_loss.item()
        
        loss = 0.5 * text_loss + 0.5 * speech_loss

        return {
            'loss': loss,
            'loss_dict': loss_dict, 
            'text_logits': text_logits,
            'speech_logits': speech_logits,
            'speech_y_pred': speech_y_pred,
            'hidden_states': hidden_states,
            'past_key_values': transformer_outputs.past_key_values,
        }


    def _top_p_filtering(self, logits, top_p=0.9, filter_value=-float('Inf')):
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[..., indices_to_remove] = filter_value
        return logits
    

    def _apply_repetition_penalty(self, logits, generated_tokens, penalty):
        for token in set(generated_tokens.view(-1)):
            logits[:, token] /= penalty
        return logits

    
    def prepare_tts_text_input_ids(self, plain_text_for_tts, prefix_token_to_wrap=[], suffix_token_to_wrap=[]):
        # assert hasattr(self, "llm_tokenizer"), f"please register llm tokenizer first!"
        llm_text_token_ids = self.llm_tokenizer(plain_text_for_tts, add_special_tokens=False).input_ids
        llm_text_token_ids = prefix_token_to_wrap + llm_text_token_ids + suffix_token_to_wrap
        text_input_ids = torch.tensor(llm_text_token_ids).view(1, -1)
        return text_input_ids


    # separate baseline generation from taslm generation for simplicity
    @torch.inference_mode()
    def _baseline_generate(
        self, 
        text_input_ids, 
        speech_input_ids,
        attention_mask=None,
        max_length=500, # max length 30 * 50 tokens / second
        text_top_p=0.9,
        speech_top_p=0.3,
        speech_repetition_max=5,
        temperature=1.0,
        speech_use_greedy=False,
        is_tts=False, # if set to True, the text_input_ids should come from self.prepare_tts_text_input_ids
        use_random=False, # if set to True, speech code will directly be randomly generated
        **gen_kwargs,
    ):
        """
        As for baseline generation, the text input ids may have different length from the speech input ids
        len(speech_input_ids) >> len(text_input_ids)
        for conditional generation, this could be a problem. we need to first slice out 
        speech input ids with the same length as the text condition; and then force generating the speech token till the condition ends
        conditional generation:
        text_input_ids: (1, N)
        speech_input_ids: (1, M), M >> N
        """
        if is_tts:
            generated_text_input_ids = text_input_ids[..., :1].clone()
            assert speech_input_ids.shape[-1] == 1, f"Invalid speech_input_ids for TTS, shape={speech_input_ids.shape}. shape should be (1, 1) (only bos token)"
        else:
            generated_text_input_ids = text_input_ids
        _cond_text_input_ids_len = generated_text_input_ids.size(-1)
        generated_speech_input_ids = speech_input_ids[..., :_cond_text_input_ids_len].clone()
        forced_speech_input_ids_for_cond = speech_input_ids[..., _cond_text_input_ids_len:].clone()
        print(generated_speech_input_ids, generated_speech_input_ids.shape)
        print(forced_speech_input_ids_for_cond, forced_speech_input_ids_for_cond.shape)
        print(speech_input_ids)
            # for unconditional generation, forced_speech_input_ids_for_cond should be with shape (1, 0). Then no force id will be apply. 
        past_key_values = None
        text_terminate_token_id = self.collate_fn_kwargs.get('text_eos_idx', self.config.llama_config.eos_token_id)
        print(f"will use {text_terminate_token_id} as text terminate token")
        forced_next_text_input_id = None
        initial_generated_length = generated_speech_input_ids.size(-1)
        forced_speech_cond_length = forced_speech_input_ids_for_cond.size(-1)
        print(initial_generated_length, forced_speech_cond_length, speech_input_ids.shape)
        repetition_token_and_counts = [-1, 0]
        for step in range(max_length - initial_generated_length):
            if past_key_values is None:
                current_text_input_ids = generated_text_input_ids
                current_speech_input_ids = generated_speech_input_ids
            else:
                current_text_input_ids = generated_text_input_ids[:, -1:].clone()
                current_speech_input_ids = generated_speech_input_ids[:, -1:].clone()
            
            forward_result = self.forward(
                text_input_ids=current_text_input_ids,
                text_attention_mask=attention_mask,
                speech_input_ids=current_speech_input_ids,
                speech_attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = forward_result['past_key_values']

            text_logits = forward_result['text_logits'][:, -1, :] / temperature
            # conduct top-p sampling
            text_logits = self._top_p_filtering(text_logits, top_p=text_top_p)
            if is_tts:
                next_text_input_id = text_input_ids[..., step+1:step+2]
            else:
                next_text_input_id = torch.multinomial(F.softmax(text_logits, dim=-1), num_samples=1)

            if forced_next_text_input_id is not None:
                next_text_input_id[..., :] = forced_next_text_input_id
            else:
                # forced_next_text_input_id is None,
                # check if the next_text_input_id indicates text termination
                if next_text_input_id.item() == text_terminate_token_id:
                    forced_next_text_input_id = text_terminate_token_id
                    # will force the next_text_input_id to be forced_next_text_input_id afterwards

            if step < forced_speech_cond_length:
                # use the forced speech token id
                next_speech_input_id = forced_speech_input_ids_for_cond[..., step: step+1]
                print(f"forced use cond speech_input ids, index range: {step}-{step+1}")
                print(next_speech_input_id)
            else:
                # sample out speech token
                speech_logits = forward_result['speech_logits'][:, -1, :] / temperature
                if speech_use_greedy:
                    # greedy decode
                    next_speech_input_id = speech_logits.argmax(dim=-1)
                else:
                    # conduct top-p sampling
                    speech_logits = self._top_p_filtering(speech_logits, top_p=speech_top_p)
                    next_speech_input_id = torch.multinomial(F.softmax(speech_logits, dim=-1), num_samples=1)

            # merge to generated
            generated_text_input_ids = torch.cat([generated_text_input_ids, next_text_input_id], dim=1)
            generated_speech_input_ids = torch.cat([generated_speech_input_ids, next_speech_input_id], dim=1)

            if attention_mask is not None:
                new_mask = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, new_mask], dim=1)
            
            if (next_speech_input_id.item() == self.config.speech_eos_token_id):
                # if the next_speech_input_id is speech_eos_token, break the loop and return the generated results
                break
            if (next_speech_input_id.item() == repetition_token_and_counts[0]):
                if repetition_token_and_counts[1] > speech_repetition_max:
                    print("detect repetition, break the generation loop.")
                    break
                else:
                    repetition_token_and_counts[1] += 1
            else:
                repetition_token_and_counts[0] = next_speech_input_id.item()
                repetition_token_and_counts[1] = 1
        
        torch.cuda.empty_cache()
        return {
            'generated_text_ids': generated_text_input_ids,
            'generated_speech_ids': generated_speech_input_ids
        }


    # custom generate function for TASLM decoding. only batch_size=1 is supported currently.
    @torch.inference_mode()
    def generate(
        self, 
        text_input_ids, 
        speech_input_ids,
        attention_mask=None,
        max_length=50,
        text_top_p=0.9,
        speech_top_p=0.3, 
        temperature=1.0,
        speech_use_greedy=False,
        is_tts=False, # if set to True, the text_input_ids should come from self.prepare_tts_text_input_ids
        use_random=False, # if set to True, speech code will directly be randomly generated
        **gen_kwargs,
    ):
        self.eval()
        if self.speech_token_type == "s3":
            return self._baseline_generate(
                text_input_ids, 
                speech_input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                text_top_p=text_top_p,
                speech_top_p=speech_top_p, 
                temperature=temperature,
                speech_use_greedy=speech_use_greedy,
                is_tts=is_tts, # if set to True, the text_input_ids should come from self.prepare_tts_text_input_ids
                use_random=use_random, # if set to True, speech code will directly be randomly generated
                **gen_kwargs,
            )
        if not hasattr(self, 'is_word_start_dict'):
            self.register_llm_word_start_tokens()
        
        if is_tts:
            generated_text_input_ids = text_input_ids[..., :1].clone()
        else:
            generated_text_input_ids = text_input_ids
        generated_speech_input_ids = speech_input_ids
        past_key_values = None
        _map_orig_eos_to_special_token_id = self.collate_fn_kwargs.get('map_orig_eos_to_special_token_id', self.config.llama_config.eos_token_id)
        forced_next_text_input_id = None
        for step in range(max_length):
            if past_key_values is None:
                current_text_input_ids = generated_text_input_ids
                current_speech_input_ids = generated_speech_input_ids
            else:
                current_text_input_ids = generated_text_input_ids[:, -1:].clone()
                current_speech_input_ids = generated_speech_input_ids[:, -1:, ...].clone()
            
            forward_result = self.forward(
                text_input_ids=current_text_input_ids,
                text_attention_mask=attention_mask,
                speech_input_ids=current_speech_input_ids,
                speech_attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = forward_result['past_key_values']

            text_logits = forward_result['text_logits'][:, -1, :] / temperature
            # conduct top-p sampling
            text_logits = self._top_p_filtering(text_logits, top_p=text_top_p)
            if is_tts:
                next_text_input_id = text_input_ids[..., step+1:step+2]
            else:
                next_text_input_id = torch.multinomial(F.softmax(text_logits, dim=-1), num_samples=1)
            if forced_next_text_input_id is not None:
                next_text_input_id[..., :] = forced_next_text_input_id

            if self.is_word_start_dict[next_text_input_id.item()]:
                # need to determine whether to switch speech token input ids
                if self.speech_token_adopt_latent_sampling:
                    speech_y_pred = forward_result['speech_y_pred'][:, -1, :]
                    _quantized, next_speech_input_ids, _loss = self.speech_embed_tokens.rvq(speech_y_pred)
                    next_speech_input_ids = next_speech_input_ids.unsqueeze(1)
                    # calculate L_reg
                    # shift_speech_y_target = _encoded_speech[..., 1:, :]
                    # shift_speech_y_pred = speech_
                else:
                    next_speech_input_ids_list = []
                    speech_logits = forward_result['speech_logits'][:, -1, :] / temperature
                    for i in range(self.speech_num_channels):
                        _channel_start_idx, _channel_end_idx = self.config.speech_vocab_size * i, self.config.speech_vocab_size * (i+1)
                        speech_logits_i = speech_logits[..., _channel_start_idx:_channel_end_idx]
                        speech_logits_i = self._top_p_filtering(speech_logits_i, top_p=speech_top_p)
                        if use_random:
                            speech_input_i = torch.randint(0, self.config.speech_vocab_size, (1, 1), device=speech_logits_i.device)
                        elif speech_use_greedy:
                            speech_input_i = speech_logits_i.argmax(dim=-1, keepdim=True)
                        else:
                            speech_input_i = torch.multinomial(F.softmax(speech_logits_i, dim=-1), num_samples=1)
                        next_speech_input_ids_list.append(speech_input_i)
                    if self.speech_num_channels > 1:
                        next_speech_input_ids = torch.cat(next_speech_input_ids_list, dim=-1)
                        next_speech_input_ids = next_speech_input_ids.unsqueeze(1)
                    else:
                        next_speech_input_ids = next_speech_input_ids_list[0] # squeeze the last dim
            else:
                next_speech_input_ids = current_speech_input_ids[:, -1:, ...].clone()

            # merge to generated
            generated_text_input_ids = torch.cat([generated_text_input_ids, next_text_input_id], dim=1)
            generated_speech_input_ids = torch.cat([generated_speech_input_ids, next_speech_input_ids], dim=1)

            if attention_mask is not None:
                new_mask = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, new_mask], dim=1)
            
            if (next_text_input_id.item() == self.config.llama_config.eos_token_id):
                # ensure the second last is the special token before breaking the loop
                if generated_text_input_ids[:, -2] == _map_orig_eos_to_special_token_id:
                    break
                else:
                    generated_text_input_ids[:, -1] = _map_orig_eos_to_special_token_id
                    forced_next_text_input_id = self.config.llama_config.eos_token_id
            if (next_text_input_id.item() == _map_orig_eos_to_special_token_id):
                forced_next_text_input_id = self.config.llama_config.eos_token_id
        
        torch.cuda.empty_cache()
        return {
            'generated_text_ids': generated_text_input_ids,
            'generated_speech_ids': generated_speech_input_ids
        }





