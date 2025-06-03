from cosyvoice.audio.audio_encoder import SenseVoiceAudioEncoder
from cosyvoice.audio.audio_segmenter import HeuristicSegmenter, SENSE_VOICE_STRIDE_SECS, SAMPLE_RATE
from cosyvoice.audio.audio_quantizer import BaseAudioQuantizer
from cosyvoice.dataset.dataset import Dataset
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from hyperpyyaml import load_hyperpyyaml
from typing import Optional

class BaseAudioTokenizer(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        self.audio_encoder = None
        self.audio_segmenter = None
        self.audio_quantizer = None
    
    def forward(
        self,
        audio_features: torch.Tensor,
        audio_features_lengths: torch.Tensor,
        text_tokens_for_audio: torch.Tensor,
        text_tokens_embed_for_audio: torch.Tensor,
        text_tokens_lengths: torch.Tensor,
        asr_alignment: Optional[torch.Tensor],
        **kwargs,
    ):
        '''
        Args:
            audio_features: torch.Tensor, shape (batch_size, max_audio_len, audio_feat_dim)
            audio_features_lengths: torch.Tensor, shape (batch_size,)
            text_tokens: torch.Tensor, shape (batch_size, max_text_len)
            text_tokens_lengths: torch.Tensor, shape (batch_size,)
        '''
        pass

    def tokenize_feature(
        self,
    ):
        pass


class SenseVoiceAudioTokenizer(BaseAudioTokenizer):
    def __init__(
        self,
        audio_encoder: nn.Module = None,
        audio_segmenter: nn.Module = None,
        audio_quantizer: nn.Module = None,
        **kwargs,
    ):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.audio_segmenter = audio_segmenter
        self.audio_segmenter.text_tokenizer = getattr(self.audio_encoder, "text_tokenizer", None) # set audio_segmenter's text_tokenizer properly without modifying the hyperPyYAML config. 
        self.audio_quantizer = audio_quantizer
    
    def forward(
        self,
        audio_features: torch.Tensor,
        audio_features_lengths: torch.Tensor,
        text_tokens_for_audio: torch.Tensor,
        text_tokens_embed_for_audio: torch.Tensor,
        text_tokens_lengths: torch.Tensor,
        asr_alignment: Optional[torch.Tensor],
        **kwargs,
    ):
        '''
        Args:
            audio_features: torch.Tensor, shape (batch_size, max_audio_len, audio_feat_dim)
            audio_features_lengths: torch.Tensor, shape (batch_size,)
            text_tokens: torch.Tensor, shape (batch_size, max_text_len)
            text_tokens_lengths: torch.Tensor, shape (batch_size,)
        '''
        encoded_results = self.audio_encoder(audio_features, audio_features_lengths, **kwargs)
        segmented_results = self.audio_segmenter(encoded_results, text_tokens_for_audio, text_tokens_embed_for_audio, text_tokens_lengths, asr_alignment, **kwargs)
        quantized_results = self.audio_quantizer(segmented_results, **kwargs)

        overall_results = {
            "encoded_results": encoded_results,
            "segmented_results": segmented_results,
            "quantized_results": quantized_results,
        }

        return overall_results
        # audio_tokens, quantized_feats, quantized_feats_lengths = self.audio_quantizer(quantized_results, **kwargs) 
        # return audio_tokens, quantized_feats, quantized_feats_lengths

    def tokenize_feature(
        self,
    ):
        pass

class JointEncoderSegmenterAudioTokenizer(BaseAudioTokenizer):
    def __init__(
        self,
        audio_joint_encoder_segmenter: nn.Module = None,
        audio_quantizer: nn.Module = None,
        **kwargs,
    ):
        super().__init__()
        self.audio_joint_encoder_segmenter = audio_joint_encoder_segmenter
        self.audio_quantizer = audio_quantizer
        self.kwargs = kwargs
    
    def forward(
        self,
        audio_features: torch.Tensor,
        audio_features_lengths: torch.Tensor,
        text_tokens_for_audio: torch.Tensor,
        text_tokens_embed_for_audio: torch.Tensor,
        text_tokens_lengths: torch.Tensor,
        *args,
        **kwargs,
    ):
        '''
        Args:
            audio_features: torch.Tensor, shape (batch_size, max_audio_len, audio_feat_dim)
            audio_features_lengths: torch.Tensor, shape (batch_size,)
            text_tokens: torch.Tensor, shape (batch_size, max_text_len)
            text_tokens_lengths: torch.Tensor, shape (batch_size,)
        '''
        encoded_results, segmented_results = self.audio_joint_encoder_segmenter(
            audio_features, audio_features_lengths, 
            text_tokens_for_audio, text_tokens_embed_for_audio, text_tokens_lengths,
            *args,
            **kwargs
        )
        quantized_results = self.audio_quantizer(segmented_results, **kwargs)

        overall_results = {
            "encoded_results": encoded_results,
            "segmented_results": segmented_results,
            "quantized_results": quantized_results,
        }

        return overall_results
    
    def tokenize_feature(
        self,
    ):
        pass

def test_heuristic():
    import os
    WORK_DIR = os.getenv('WORK_DIR')
    # test sensevoice tokenizer
    audio_tokenizer_config_fpath = "./audio_tokenizer_heuristic.yaml"
    with open(audio_tokenizer_config_fpath, 'r') as f:
        print('Loading configs')
        hyper_config = load_hyperpyyaml(f)
    print(hyper_config)
    sensevoice_tokenizer = hyper_config["audio_tokenizer"]
    # wav_frontend = sensevoice_tokenizer.audio_encoder.frontend
    # print(wav_frontend)
    # print(wav_frontend.output_size())
    # sensevoice_tokenizer = SenseVoiceAudioTokenizer()
    audio_fpaths = [
        # f"{WORK_DIR}/rtslm/CosyVoice/cross_lingual.wav",
        # f"{WORK_DIR}/rtslm/CosyVoice/cross_lingual.wav",
        f"{WORK_DIR}/rtslm/CosyVoice/instruct4.wav",
        # f"{WORK_DIR}/rtslm/CosyVoice/7190_90543_000016_000000.wav",
        f"{WORK_DIR}/rtslm/CosyVoice/en.mp3",
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    sensevoice_tokenizer.to(device)
    sensevoice_tokenizer.eval()
    with torch.no_grad():
        audio_feats, audio_feats_lengths = sensevoice_tokenizer.audio_encoder.extract_feature(
            audio_fpaths,
            device=device,
            cache={},
            language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
        ) # TODO: move audio extraction to audio utils
        # print(audio_feats.shape)
        # print(audio_feats[0,4:5,:])

        overall_results = sensevoice_tokenizer(
            audio_feats,
            audio_feats_lengths,
            None,
            None,
            None,
        )
        alignments = overall_results["segmented_results"]["alignments"]
        for alignment in alignments:
            for i, (segment, (start, end)) in enumerate(alignment):
                start = round(start * (SENSE_VOICE_STRIDE_SECS * SAMPLE_RATE))
                end = round(end * (SENSE_VOICE_STRIDE_SECS * SAMPLE_RATE))
                try:
                    print("start:", start, "end:", end, "segment:", segment)
                except Exception as e:
                    print("start:", start, "end:", end, "segment:", e)
            print("-" * 50)
        segmented_results = overall_results["segmented_results"]
        segmented_feats = segmented_results["segmented_feats"]
        segmented_feats_lengths = segmented_results["segmented_feats_lengths"]
        print(segmented_feats.shape, segmented_feats_lengths)
        quantized_results = overall_results["quantized_results"]
        quantized_feats = quantized_results["quantized_feats"]
        quantized_feats_lengths = quantized_results["quantized_feats_lengths"]
        quantized_indices = quantized_results["quantized_indices"]
        print(quantized_indices)
        print(quantized_feats.shape, quantized_feats_lengths, quantized_indices.shape)

def test_cross_attn():
    import os
    WORK_DIR = os.getenv('WORK_DIR')
    # test sensevoice tokenizer
    audio_tokenizer_config_fpath = "./audio_tokenizer_cross_attn.yaml"
    with open(audio_tokenizer_config_fpath, 'r') as f:
        print('Loading configs')
        model_config = load_hyperpyyaml(f)
    print(model_config)
    audio_tokenizer = model_config['audio_tokenizer']

    dataset_config_fpath = "/proj/mtklmadm/dev/mtk53678/rtslm/CosyVoice/cosyvoice/dataset/config_for_test.yaml"
    with open(dataset_config_fpath, 'r') as f:
        print('Loading configs')
        dataset_config = load_hyperpyyaml(f)
    print(dataset_config)

    dataset_for_test = Dataset(dataset_config["test_dataset_parquet_data_list_fpath"], data_pipeline=dataset_config['data_pipeline'], mode='train', shuffle=True, partition=True)
    batch_data_for_test = None
    for d in dataset_for_test:
        batch_data_for_test = d
        break
    print(batch_data_for_test.keys())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_feat = batch_data_for_test['audio_feat'].to(device)
    audio_feat_len = batch_data_for_test['audio_feat_len'].to(device)
    text_token = batch_data_for_test['text_token'].to(device)
    text_token_len = batch_data_for_test['text_token_len'].to(device)
    if audio_tokenizer.audio_segmenter.is_word_level:
        words_begin_index = batch_data_for_test['words_begin_index'].to(device)
        words_end_index = batch_data_for_test['words_end_index'].to(device)
        words_index_len = batch_data_for_test['words_index_len'].to(device)

    target_shape_for_text_token_embed = (text_token.shape[0], text_token.shape[1], model_config['text_encoder_input_size']) # shape of fake text_token_embedding
    print(f"target shape of text token embedding: {target_shape_for_text_token_embed}")
    # generate fake text_token_embed
    text_token_fake_embed = torch.randn(target_shape_for_text_token_embed, device=device)
    audio_tokenizer.eval()
    audio_tokenizer.to(device)
    with torch.no_grad():
        overall_results = audio_tokenizer(
            audio_feat,
            audio_feat_len,
            text_token,
            text_token_fake_embed,
            text_token_len,
        )
        segmented_results = overall_results["segmented_results"]
        segmented_feats = segmented_results["segmented_feats"]
        segmented_feats_lengths = segmented_results["segmented_feats_lengths"]
        print(segmented_feats.shape, segmented_feats_lengths)
        quantized_results = overall_results["quantized_results"]
        quantized_feats = quantized_results["quantized_feats"]
        quantized_feats_lengths = quantized_results["quantized_feats_lengths"]
        quantized_indices = quantized_results["quantized_indices"]
        print(quantized_indices)
        print(quantized_feats.shape, quantized_feats_lengths, quantized_indices.shape)

if __name__ == "__main__":
    test_heuristic()
    # test_cross_attn()