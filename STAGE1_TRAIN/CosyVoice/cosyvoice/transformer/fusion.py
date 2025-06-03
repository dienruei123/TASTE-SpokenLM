# This file explore multiple fusion types between the derived audio token and the text token (encoded)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from cosyvoice.utils.common import IGNORE_ID
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

class Concat(nn.Module):
    def __init__(
        self,
        audio_first: bool = True,
        ignore_id: int = IGNORE_ID,
        skip_prefix_idx: int = 0, # set to 0 will skip no prefix audio tokens
        skip_postfix_idx: int = 0, # set to 0 will skip no postfix audio tokens
    ):
        super().__init__()
        self.audio_first = audio_first
        self.ignore_id = ignore_id
        # skip init, no submodule
        # handle prefix or postfix skipping
        assert skip_prefix_idx >= 0, f"Please set prefix_idx >= 0. 0 for no skipping, i for starting from the i-th prefix. Current: {skip_prefix_idx}"
        assert skip_postfix_idx <= 0, f"Please set postfix_idx <= 0. 0 for no skipping, -i for skipping i postfix. Current: {skip_postfix_idx}"
        self.skip_prefix_idx = skip_prefix_idx
        self.skip_postfix_idx = skip_postfix_idx
    
    def forward(
        self,
        audio_token_encoded,
        audio_token_len,
        text_token_encoded,
        text_token_len,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_token_encoded: (B, T, C)
            audio_token_len: (B,)
            text_token_encoded: (B, T, C)
        """
        assert audio_token_encoded.shape[-1] == text_token_encoded.shape[-1], "Inconsistency in feature dimension!"
        unpad_audio_token = unpad_sequence(audio_token_encoded, audio_token_len.cpu(), batch_first=True)
        unpad_text_token = unpad_sequence(text_token_encoded, text_token_len.cpu(), batch_first=True)
        for i in range(len(unpad_audio_token)):
            if self.skip_postfix_idx < 0:
                unpad_audio_token[i] = unpad_audio_token[i][self.skip_prefix_idx: self.skip_postfix_idx]
            else:
                unpad_audio_token[i] = unpad_audio_token[i][self.skip_prefix_idx:]
        # print(unpad_audio_token[0].shape, audio_token_len)
        # assert False, "stop for debug"
        if self.audio_first:
            audio_text_token_encoded = [torch.concat([unpad_audio_token[i], unpad_text_token[i]], dim=0) for i in range(len(audio_token_len))]
        else:
            audio_text_token_encoded = [torch.concat([unpad_text_token[i], unpad_audio_token[i]], dim=0) for i in range(len(text_token_len))]
        audio_text_token_len = torch.tensor([i.size(0) for i in audio_text_token_encoded], dtype=torch.int32)
        audio_text_token_encoded = pad_sequence(audio_text_token_encoded, batch_first=True, padding_value=self.ignore_id)
        return audio_text_token_encoded, audio_text_token_len


# weighted sum
class WeightedSum(nn.Module):
    def __init__(
        self,
        ignore_id: int = IGNORE_ID,
        normalize: bool = True,
        use_trainable_weight: bool = True,
        use_layer_norm: bool = False,
        weight_init_type: str = 'balance'
    ):
        super().__init__()
        if weight_init_type == 'balance':
            self.weights = nn.Parameter(torch.tensor([1., 1.]), requires_grad=use_trainable_weight)
        elif weight_init_type == 'zero_audio':
            self.weights = nn.Parameter(torch.tensor([-2., 2.]), requires_grad=use_trainable_weight)
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            # TODO: Maybe we can add layernorm here. Currently I think it is not necessary. 
            raise NotImplementedError
        self.normalize = normalize
        self.ignore_id = IGNORE_ID
    
    def forward(
        self,
        audio_token_encoded,
        audio_token_len,
        text_token_encoded,
        text_token_len,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_token_encoded: (B, T, C)
            audio_token_len: (B,)
            text_token_encoded: (B, T, C)
            audio_token_len: (B,)
        """
        assert audio_token_encoded.shape[-1] == text_token_encoded.shape[-1], "Inconsistency in feature  dimension!"
        assert audio_token_encoded.shape[-2] == text_token_encoded.shape[-2], "Inconsistency in sequence dimension!"
        assert (audio_token_len == text_token_len).sum().item() == audio_token_len.size(0), "Find length mismatch"
        # unpad_audio_token = unpad_sequence(audio_token_encoded, audio_token_len.cpu(), batch_first=True)
        # unpad_text_token = unpad_sequence(text_token_encoded, text_token_len.cpu(), batch_first=True)
        # use softmax to prevent negative weight
        weights = F.softmax(self.weights, dim=0)
        weights = weights.view(2, 1, 1, 1)

        if self.normalize:
            audio_token_encoded = F.layer_norm(
                audio_token_encoded,
                (audio_token_encoded.size(-1),)
            )
            text_token_encoded = F.layer_norm(
                text_token_encoded,
                (text_token_encoded.size(-1),)
            )
        inputs = torch.stack([audio_token_encoded, text_token_encoded], dim=0)
        audio_text_token_encoded = (weights * inputs).sum(dim=0)
        audio_text_token_len = audio_token_len
        # audio_text_token_len = torch.tensor([i.size(0) for i in audio_text_token_encoded], dtype=torch.int32)
        return audio_text_token_encoded, audio_text_token_len


def test_weighted_sum():
    fusion_layer = WeightedSum(normalize=False)
    x = torch.randn((2, 10, 100))
    x_len = torch.tensor([5, 10])
    y = torch.randn((2, 10, 100))
    y_len = torch.tensor([5, 10])
    z, z_len = fusion_layer(
        x, 
        x_len,
        y,
        y_len
    )
    print(z.shape, z_len)
    print(x[0][0])
    print(y[0][0])
    print(z[0][0])
    print((x[0][0] + y[0][0]) / 2)

if __name__ == "__main__":
    test_weighted_sum()