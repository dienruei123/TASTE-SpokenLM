from typing import Dict, Optional, Union
import torch
import logging
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from cosyvoice.utils.mask import subsequent_mask, make_pad_mask
from cosyvoice.utils.model_utils import load_whisper_whole_model
from transformers import WhisperTokenizer

SENSE_VOICE_STRIDE_SECS = 0.06
SAMPLE_RATE = 16000

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BaseSegmenter(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        encoded_results: Dict[str, Optional[torch.Tensor]],
        **kwargs,
    ):
        raise NotImplementedError

class HeuristicSegmenter(BaseSegmenter):
    def __init__(
        self,
        text_tokenizer=None
    ):
        super().__init__()
        self.text_tokenizer = text_tokenizer
    
    def _get_single_alignment(
        self,
        tokens,
        window_size=9, 
        num_prev_elements=4,
        **kwargs,
    ):
        assert window_size % 2 == 1, "window_size must be odd"

        # get center index
        left_map = []
        right_map = []
        for i in range(num_prev_elements, len(tokens)):
            token = tokens[i]
            if len(left_map) == len(right_map):
                left_map.append([token, i])
            elif token != left_map[-1][0]:
                right_map.append([tokens[i-1], i-1])
                left_map.append([token, i])
        if len(left_map) != len(right_map):
            right_map.append([tokens[-1], len(tokens)-1])
        center_map = [(token, (left + right) // 2) for (token, left), (_, right) in zip(left_map, right_map)
                    if token != '<unk>']

        if len(center_map) == 0:
            return []

        # merging
        def _limit_left(index):
            return max(index, num_prev_elements)
        def _limit_right(index):
            return min(index, len(tokens))

        r = (window_size - 1) // 2

        range_map = [
            [
                center_map[0][0],
                [_limit_left(center_map[0][1] - r,), _limit_right(center_map[0][1] + r + 1)]
            ]
        ]
        i = 1
        while i < len(center_map):
            token, index = center_map[i]
            if not token.startswith('▁'):
                range_map[-1][0] += token
                range_map[-1][1][1] = _limit_right(center_map[i][1] + r + 1)
            else:
                range_map.append(
                    [
                        token,
                        [_limit_left(center_map[i][1] - r,), _limit_right(center_map[i][1] + r + 1)]
                    ])
            i += 1

        alignments = [(segment, (left - num_prev_elements, right - num_prev_elements))
                    for segment, (left, right) in range_map]
        return alignments
    
    def get_alignment(
        self,
        ctc_logits: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        text_tokenizer,
        window_size=9, 
        num_prev_elements=4,
        **kwargs,
    ):  
        bsz = ctc_logits.size(0)

        alignments = []
        for i in range(bsz):
            x = ctc_logits[i, : encoder_out_lens[i].item(), :]
            yseq = x.argmax(dim=-1)
            # preds.append(yseq.tolist())
            tokens = text_tokenizer.sp.id_to_piece(yseq.detach().cpu().numpy().tolist())
            # print(tokens) # please ensure LANG=en_US.UTF-8 is set properly
            alignment = self._get_single_alignment(tokens, window_size=window_size, num_prev_elements=num_prev_elements)
            alignments.append(alignment)
        return alignments

    def forward(
        self,
        encoded_results: Dict[str, Optional[torch.Tensor]],
        *args,
        **kwargs,
    ):
        '''
        Args:
            extracted_results: Dict[str, Optional[torch.Tensor]], extracted results from audio extractor
                last_hidden_states: torch.Tensor, shape (batch_size, max_audio_len, hidden_size)
                last_hidden_states_lengths: torch.Tensor, shape (batch_size,)
                ctc_logits: torch.Tensor, shape (batch_size, max_audio_len, vocab_size)
        '''
        ctc_logits = encoded_results['ctc_logits']
        encoded_feats = encoded_results['encoded_feats']
        encoded_feats_lengths = encoded_results['encoded_feats_lengths']
        alignments = self.get_alignment(ctc_logits, encoded_feats_lengths, text_tokenizer=self.text_tokenizer)

        segmented_feats, segmented_feats_lengths = [], torch.zeros(encoded_feats_lengths.shape, dtype=encoded_feats_lengths.dtype, device=encoded_feats_lengths.device)
        for i, (alignment, encoded_feat, encoded_feat_length) in enumerate(zip(alignments, encoded_feats, encoded_feats_lengths)):
            # print(alignment)
            # alignment: (segment, (start, end))
            # encoded_feat: (max_audio_len, hidden_size)
            # encoded_feat_length: (1, )
            new_segment_feat_len = len(alignment)
            if new_segment_feat_len == 0:
                segmented_feats.append(torch.zeros((1, encoded_feat.size(1)), dtype=encoded_feat.dtype, device=encoded_feat.device))
                segmented_feats_lengths[i] = 1
                continue

            new_segment_feat = torch.zeros((new_segment_feat_len, encoded_feat.size(1)), dtype=encoded_feat.dtype, device=encoded_feat.device)
            for j, (_segment, (start, end)) in enumerate(alignment):
                assert start < end and end <= encoded_feat_length.item(), f"start: {start}, end: {end}, encoded_feat_length: {encoded_feat_length}"
                new_segment_feat[j, :] = encoded_feat[start:end, :].mean(dim=0)
            segmented_feats.append(new_segment_feat)
            segmented_feats_lengths[i] = new_segment_feat_len
            # TODO: implement segmentation based on alignment
        segmented_feats = pad_sequence(segmented_feats, batch_first=True)
        
        
        results = {
            'alignments': alignments,
            'segmented_feats': segmented_feats, # TODO: segmented_feats
            'segmented_feats_lengths': segmented_feats_lengths, # TODO: segmented_feats_lengths
        }
        return results


# TODO: CrossAttentionSegmenter

class CrossAttentionSegmenter(torch.nn.Module):
    def __init__(
        self,
        is_word_level: bool,
        audio_decoder: torch.nn.Module,
    ):
        super().__init__()
        self.is_word_level = is_word_level
        self.audio_decoder = audio_decoder
    
    def subword_to_word_text_embed(
        self, 
    ):
        pass
    
    def forward(
        self,
        encoded_results: Dict[str, Optional[torch.Tensor]],
        text_token_for_audio: Optional[torch.Tensor],
        text_token_embed_for_audio: Optional[torch.Tensor], 
        text_token_len: Optional[torch.Tensor],
        *args,
        words_index: Dict[str, Optional[torch.Tensor]] = None, # currently not supported
        **kwargs,
    ):
        """
        args:
            audio_features: (B, T, C)
            audio_features_lengths: (B,)
            text_token_embed_for_audio: (B, T, C)
            text_token_len: (B,)
        kwargs:
            words_index: None or {
                'words_begin_index': words_begin_index,
                'words_end_index': words_end_index,
                'words_index_len': words_index_len,
            }
            (refers to llm.audio_llm.py)
        """
        # in cross-attention, we treat audio features as memory feat
        audio_feats, audio_feats_len = encoded_results['encoded_feats'], encoded_results['encoded_feats_lengths']
        audio_feats_mask = ~make_pad_mask(audio_feats_len, audio_feats.size(1)).unsqueeze(1)
        # logger.debug(f"{audio_feats.shape}, {audio_feats_len.shape}, {audio_feats_mask.shape}")
        # logger.debug(f"{text_token_embed_for_audio.shape}, {text_token_len.shape}")
        segmented_feats, _, total_lengths = self.audio_decoder(
            audio_feats,
            audio_feats_mask,
            text_token_embed_for_audio,
            text_token_len,
        )
        segmented_feats_lengths = text_token_len
        # logger.debug(f"{segmented_feats.shape}, {segmented_feats_lengths.shape}")
        

        segmented_results = {
            'segmented_feats': segmented_feats, 
            'segmented_feats_lengths': segmented_feats_lengths, 
        }

        return segmented_results

class WhisperCrossAttentionSegmenter(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str, # NOTE: We need this for getting the tokenizer.
        decoder_model: nn.Module = None,
        attn_implementation: str = "eager",
        dtype: str = "float32",
        **kwargs,
    ):
        super().__init__()
        if decoder_model == None:
            whole_model, _torch_dtype = load_whisper_whole_model(
                model_name_or_path,
                attn_implementation=attn_implementation,
                dtype=dtype,
            )
            self.decoder = whole_model.get_decoder()
        else:
            self.decoder = decoder_model
        # print(self.decoder)
        self.model_name_or_path = model_name_or_path
        self.tokenizer = WhisperTokenizer.from_pretrained(
            model_name_or_path,
        )
    
    def forward(
        self,
        encoded_results: Dict[str, Optional[torch.Tensor]],
        text_token_for_audio: Optional[torch.Tensor],
        text_token_embed_for_audio: Optional[torch.Tensor], 
        text_token_len: Optional[torch.Tensor],
        *args,
        whisper_text_token: Optional[torch.Tensor] = None, # whisper text tokens with decoder prefix
        whisper_text_token_len: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        args:
            audio_features: (B, T, C)
            audio_features_lengths: (B,)
            text_token_embed_for_audio: (B, T, C)
            text_token_len: (B,)
        kwargs:
            words_index: None or {
                'words_begin_index': words_begin_index,
                'words_end_index': words_end_index,
                'words_index_len': words_index_len,
            }
            (refers to llm.audio_llm.py)
        """
        pass
        
        
def test_whisper_cross_attn_segmenter():
    model_name_or_path = "/proj/mtklmadm/dev/mtk53678/rtslm_storage/pretrained_models/distil-whisper-large-v3"
    cross_attn_segmenter = WhisperCrossAttentionSegmenter(
        model_name_or_path,
        attn_implementation = "flash_attention_2",
        dtype = "bfloat16",
    )
    print(cross_attn_segmenter.decoder_input_ids)


class StraightThroughSegmenter(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
    
    def forward(
        self,
        encoded_results: Dict[str, Optional[torch.Tensor]],
        *args,
        **kwargs,
    ):
        segmented_feats, segmented_feats_lengths = encoded_results["encoded_feats"], encoded_results["encoded_feats_lengths"]

        segmented_results = {
            'segmented_feats': segmented_feats,
            'segmented_feats_lengths': segmented_feats_lengths,
        }

        return segmented_results


class WholeAveragePoolingSegmenter(torch.nn.Module):
    def __init__(
        self,
        expand_to_token_len=True
    ):
        super().__init__()
        self.expand_to_token_len = expand_to_token_len

    def forward(
        self,
        encoded_results: Dict[str, Optional[torch.Tensor]],
        text_token_for_audio: Optional[torch.Tensor],
        text_token_embed_for_audio: Optional[torch.Tensor], 
        text_token_len: Optional[torch.Tensor],
        *args,
        **kwargs,
    ):
        audio_feats, audio_feats_len = encoded_results['encoded_feats'], encoded_results['encoded_feats_lengths']
        audio_feats_mask = ~ make_pad_mask(audio_feats_len, audio_feats.size(1)).unsqueeze(-1)
        single_embed = ((audio_feats * audio_feats_mask).sum(dim=-2, keepdim=True) / audio_feats_mask.sum(dim=-2, keepdim=True))

        if self.expand_to_token_len:
            segmented_results = {
                'segmented_feats': single_embed.expand(-1, text_token_len.max(), -1),
                'segmented_feats_lengths': text_token_len,
            }
        else:
            segmented_results = {
                'segmented_feats': single_embed,
                'segmented_feats_lengths': torch.ones(single_embed.size(0)).to(single_embed.device),
            }

        return segmented_results


def test_WholeAveragePoolingSegmenter():
    seg = WholeAveragePoolingSegmenter(expand_to_token_len=False)
    text_token_len = torch.tensor([1, 2])
    encoded_results = {
        "encoded_feats": torch.randint(high=10, size=(2, 5, 3)) * 0.1,
        "encoded_feats_lengths": torch.tensor([2, 5])
    }
    print(encoded_results)
    results = seg(encoded_results, None, None, text_token_len)
    print(results)


class LocalAveragePoolingSegmenter(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        encoded_results: Dict[str, Optional[torch.Tensor]],
        text_token_for_audio: Optional[torch.Tensor],
        text_token_embed_for_audio: Optional[torch.Tensor], 
        text_token_len: Optional[torch.Tensor],
        asr_alignment,
        *args,
        **kwargs,
    ):
        audio_feats = encoded_results['encoded_feats'] # [batch, audio_feat_len, audio_embed]
        audio_feats_len = encoded_results['encoded_feats_lengths'] # [batch]
        # asr_alignment: [batch, text_token_len, start_and_end]

        indexes = torch.arange(audio_feats.size(1)).repeat(asr_alignment.size(0) * asr_alignment.size(1)) \
            .reshape(asr_alignment.size(0), asr_alignment.size(1), -1).to(audio_feats.device)
        token_mask = ~ make_pad_mask(text_token_len, text_token_for_audio.size(1)).unsqueeze(-1)
        
        mask = (token_mask & (indexes >= asr_alignment[:, :, 0:1]) &  (indexes <= asr_alignment[:, :, 1:2])).unsqueeze(-1)
        # [batch, text_token_len, audio_feat_len, 1]
        
        expanded_audio_feats = audio_feats.unsqueeze(1).expand(-1, mask.size(1), -1, -1)  
        # [batch, 1, audio_feat_len, audio_embed]

        segmented_feats = ((expanded_audio_feats * mask).sum(dim=-2) / mask.sum(dim=-2))
        segmented_feats = torch.nan_to_num(segmented_feats)

        segmented_results = {
            'segmented_feats': segmented_feats,  # [batch, text_token_len, embed]
            'segmented_feats_lengths': text_token_len,
        }

        return segmented_results

def test_LocalAveragePoolingSegmenter():
    seg = LocalAveragePoolingSegmenter()
    text_token_for_audio = torch.tensor([[299, 0], [387, 323]])
    text_token_len = torch.tensor([1, 2])
    asr_alignment = torch.tensor([[[0, 1], [0, 0]], [[0, 3], [3, 3]]])
    encoded_results = {
        "encoded_feats": torch.randint(high=10, size=(2, 5, 3)) * 0.1,
        "encoded_feats_lengths": torch.tensor([2, 5]),
    }
    print(encoded_results)
    results = seg(encoded_results, text_token_for_audio, None, text_token_len, asr_alignment)
    print(results)


if __name__ == "__main__":
    test_whisper_cross_attn_segmenter()