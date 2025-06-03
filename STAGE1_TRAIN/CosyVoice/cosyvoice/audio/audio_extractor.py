import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Optional
from transformers import WhisperProcessor

class WhisperAudioExtractor(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        pad_to_whisper_input_size: bool = True, # will automatically pad to 30 sec input
        use_orig_length: bool = True, # use original audio length 
    ):
        super().__init__()
        self.target_sample_rate = 16_000
        self.expected_seq_length = 3000
        self.pad_to_whisper_input_size = pad_to_whisper_input_size
        self.use_orig_length = use_orig_length
        self.processor = WhisperProcessor.from_pretrained(model_name_or_path)

    def extract_feature(
        self,
        audio_fpaths: List[str],
        pad_to_whisper_input_size: Optional[bool] = None,
        **cfg,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_feat_list, audio_feat_len_list = [], []
        for audio_fpath in audio_fpaths:
            waveform, sr = librosa.load(audio_fpath, sr=16_000, mono=True) # target sr is 16_000 for whisper
            inputs = self.processor(waveform, sampling_rate=16000, return_tensors='pt', max_length=waveform.shape[-1])
            audio_feat = inputs["input_features"].transpose(-1, -2).squeeze(0) # (B, C, T) -> (T, C)
            audio_feat_list.append(audio_feat)
            audio_feat_len_list.append(audio_feat.shape[0])
        if pad_to_whisper_input_size:
            audio_feat = self.pad_to_whisper_input_size(audio_feat_list)
        else:
            audio_feat = pad_sequence(audio_feat_list, batch_first=True)
        audio_feat_len = torch.tensor(audio_feat_len_list, dtype=torch.int32)

        return audio_feat, audio_feat_len

    def pad_to_whisper_input(self, audio_feat: List[torch.Tensor], padding_value=0.0):
        b, t = len(audio_feat), self.expected_seq_length
        c = audio_feat[0].shape[-1] # each tensor is with shape (T, C)
        padded_tensors = torch.full(
            (b, t, c),
            padding_value,
        )
        for i, tensor in enumerate(audio_feat):
            length = tensor.size(0)
            padded_tensors[i, :length] = tensor
        
        return padded_tensors

    def forward(
        self,
        input: torch.Tensor, # (B, T)
        input_lengths: List[int],
        **kwargs,
    ):  
        batch_size = input.size(0)
        feats = []
        feats_lens = []
        for i in range(batch_size):
            waveform_length = input_lengths[i]
            waveform = input[i][:waveform_length]
            audio_feat = self.processor(waveform, sampling_rate=16000, return_tensors='pt', max_length=waveform_length)
            audio_feat = audio_feat["input_features"].transpose(-1, -2).squeeze(0) # (B, C, T) -> (T, C)
            audio_feat_len = len(audio_feat)
            feats.append(audio_feat)
            feats_lens.append(audio_feat_len)
        if self.pad_to_whisper_input_size:
            feats = self.pad_to_whisper_input(feats)
        else: 
            feats = pad_sequence(feats, batch_first=True, padding_value=0.0)
        feats_lens = torch.tensor(feats_lens, dtype=torch.int32) if self.use_orig_length else torch.tensor([self.expected_seq_length] * len(feats), dtype=torch.int32)
        return feats, feats_lens

