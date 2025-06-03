# pip install vector-quantize-pytorch
from typing import Dict, Optional, Union, List, Tuple
import torch
import torch.nn as nn
from .vq.vector_quantize_pytorch import VectorQuantize, lens_to_mask
from .vq.residual_vq import ResidualVQ

class BaseAudioQuantizer(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        heads,
        decay,
        kmeans_init,
        kmeans_iters,
        codebook_dim = None,
        **kwargs,
    ):
        super().__init__()
        self.vq = VectorQuantize(
            dim=dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            decay=decay,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            heads=heads,
            **kwargs,
        )

    def forward(
        self,
        segmented_results: Dict[str, Optional[torch.Tensor]],
        **kwargs,
    ):
        '''
        Args:
            segmented_results: 
                segmented_feats: torch.Tensor, shape (batch_size, max_audio_len, audio_feat_dim)
                segmented_feats_lengths: torch.Tensor, shape (batch_size,)
        '''
        segmented_feats, segmented_feats_lengths = segmented_results["segmented_feats"], segmented_results["segmented_feats_lengths"]
        mask = lens_to_mask(segmented_feats_lengths, segmented_feats.shape[1])
        quantized, indices, commit_loss = self.vq(segmented_feats, mask=mask)

        results = {
            'quantized_feats': quantized,
            'quantized_feats_lengths': segmented_feats_lengths,
            'quantized_indices': indices,
            'commit_loss': commit_loss,
        }

        return results


class StraightThroughAudioQuantizer(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
    
    def forward(
        self,
        segmented_results: Dict[str, Optional[torch.Tensor]],
        **kwargs,
    ):
        segmented_feats, segmented_feats_lengths = segmented_results["segmented_feats"], segmented_results["segmented_feats_lengths"]

        results = {
            'quantized_feats': segmented_feats,
            'quantized_feats_lengths': segmented_feats_lengths,
            'quantized_indices': None,
        }

        return results

class RVQAudioQuantizer(nn.Module):
    def __init__(
        self,
        dim = 1280, # input_dim
        num_quantizers = 4, # residual depth
        codebook_dim = None, # codebook dim, whether to down-project from dim --> codebook_dim or not, default will set to dim
        quantize_dropout = False, # set to true to allow raw input flowing through
        # rvq kwargs related to vq
        kmeans_init = True, 
        codebook_size = 256,
        decay = 0.99, 
        **vq_kwargs,
    ):  
        super().__init__()
        self.rvq = ResidualVQ(
            dim = dim, 
            num_quantizers = num_quantizers,
            codebook_dim = codebook_dim,
            quantize_dropout = quantize_dropout,
            # vq-related
            kmeans_init = kmeans_init,
            codebook_size = codebook_size,
            decay = decay,
            **vq_kwargs,
        )
        self.dim = dim
        self.pad_index = 0
        self.ignore_index = -1
    
    def forward(
        self,
        segmented_results: Dict[str, Optional[torch.Tensor]],
        **kwargs,
    ):
        segmented_feats, segmented_feats_lengths = segmented_results["segmented_feats"], segmented_results["segmented_feats_lengths"]
        mask = lens_to_mask(segmented_feats_lengths, segmented_feats.shape[1])
        quantized, indices, commit_loss = self.rvq(segmented_feats, mask=mask)
        # print(commit_loss) # should be with shape: (num_quantizers,)
        sum_commit_loss_rvq = commit_loss.sum() # sum over the loss for the total one
        quantized_results = {
            'quantized_feats': quantized,
            'quantized_feats_lengths': segmented_feats_lengths,
            'quantized_indices': indices,
            'quantized_loss': sum_commit_loss_rvq,
        }

        return quantized_results
    
    @torch.no_grad()
    def encode(
        self,
        indices,
        indices_lengths,
        apply_mask=False,
    ):
        # mask = lens_to_mask(indices_lengths, indices.shape[1])
        if apply_mask:
            # self.pad_vector = self.pad_vector.to(indices.device)
            # self.ignore_vector = self.ignore_vector.to(indices.device)
            _bsz, _tsz, _qsz = indices.shape
            _pad_mask = (indices == self.pad_index).sum(-1) == _qsz
            _ignore_mask = (indices == self.ignore_index).sum(-1) == _qsz
            to_calculate_mask = ~torch.logical_or(_pad_mask, _ignore_mask)
            to_calculate_indices = indices[to_calculate_mask]
            _quantized_feats = self.rvq.get_output_from_indices(indices[to_calculate_mask])
            _new_quantized_feats = torch.zeros((_bsz, _tsz, self.dim), device=_quantized_feats.device, dtype=_quantized_feats.dtype)
            _new_quantized_feats[to_calculate_mask] = _quantized_feats
            quantized_feats = _new_quantized_feats
        else:
            quantized_feats = self.rvq.get_output_from_indices(indices)
        quantized_results = {
            'quantized_feats': quantized_feats,
            'quantized_feats_lengths': indices_lengths,
        }

        return quantized_results


