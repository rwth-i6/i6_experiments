"""
Multi-codebook frozen random-projection quantizer for BEST-RQ.

Combines:
* BEST-RQ (Chiu et al. 2022, arXiv:2202.01855): a *frozen* random projection + *frozen*
  random codebook produce discrete targets; both the projected vector and the codebook
  entries are **L2-normalized** before the nearest-neighbour lookup (critical for uniform
  codebook utilization). Input features are stacked by the encoder subsampling factor so
  that one target corresponds to one encoder output frame.
* Multi-codebook (Baumann et al. 2025, arXiv:2501.16131): N *independent* frozen
  projections + codebooks produce N target streams; the SSL loss averages the per-codebook
  cross-entropies. N=6 was best there; we default to N=4 (paper-faithful codebook size).

Design notes (lessons from the broken in-repo implementations we reviewed):
* projection matrices and codebooks are registered as **buffers** (not Parameters) and the
  whole forward is under ``@torch.no_grad()`` -> they never receive gradients, so the
  targets stay fixed for the entire run.
* the nearest code is found via the normalized dot product (== cosine similarity ==
  argmin L2 between unit vectors), computed one codebook at a time to bound peak memory
  (no [B, T, N, V] materialization).
* everything runs in float32 regardless of AMP, so the argmax is not perturbed by bf16.
* this module does NOT normalize the input -- the caller must pass already input-normalized
  features (see ``input_norm.masked_mean_var_norm``), exactly the same tensor that (after
  masking) feeds the encoder.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class MultiCodebookRandomProjectionQuantizer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        stack_size: int,
        codebook_dim: int,
        vocab_size: int,
        num_codebooks: int,
        seed: int = 42,
    ):
        """
        :param input_dim: per-frame feature dim F (e.g. 80 log-mel).
        :param stack_size: number of consecutive frames stacked per target (= encoder time
            subsampling factor, e.g. 4). Projection input dim is ``input_dim * stack_size``.
        :param codebook_dim: projected / code vector dim (BEST-RQ ``h``, e.g. 16).
        :param vocab_size: number of code vectors per codebook (e.g. 8192).
        :param num_codebooks: number of independent codebooks N (e.g. 4).
        :param seed: RNG seed for the (frozen) random projection + codebook init. Fixed so the
            targets are reproducible and identical across DDP ranks / checkpoints.
        """
        super().__init__()
        self.input_dim = input_dim
        self.stack_size = stack_size
        self.codebook_dim = codebook_dim
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks
        self.seed = seed
        self.proj_in = input_dim * stack_size

        gen = torch.Generator().manual_seed(seed)
        # Random projection: Gaussian (paper init). Shape [N, proj_in, codebook_dim].
        projection = torch.randn(num_codebooks, self.proj_in, codebook_dim, generator=gen)
        # Codebook: standard normal, then L2-normalized once (kept normalized as a buffer).
        # Shape [N, vocab_size, codebook_dim].
        codebook = torch.randn(num_codebooks, vocab_size, codebook_dim, generator=gen)
        codebook = torch.nn.functional.normalize(codebook, dim=-1)

        # Buffers (not Parameters): saved in state_dict, moved by .to(device), never optimized.
        self.register_buffer("projection", projection, persistent=True)
        self.register_buffer("codebook", codebook, persistent=True)

    @torch.no_grad()
    def forward(
        self, features: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        :param features: [B, T, F] input-normalized features (same tensor that, masked, feeds
            the encoder). F must equal ``input_dim``.
        :param lengths: [B] valid frame counts at the input (pre-subsampling) rate, optional.
        :return: (targets, target_lengths) where
            targets: [B, T2, N] int64 code indices, T2 = T // stack_size;
            target_lengths: [B] = lengths // stack_size (or None if lengths is None).
        """
        assert features.dim() == 3 and features.shape[-1] == self.input_dim, (
            f"expected [B,T,{self.input_dim}], got {tuple(features.shape)}"
        )
        b, t, f = features.shape
        t2 = t // self.stack_size
        # Compute targets in float32 so the argmax is unaffected by AMP/bf16.
        x = features[:, : t2 * self.stack_size, :].reshape(b, t2, self.proj_in).float()  # [B, T2, P]

        targets = []
        for n in range(self.num_codebooks):
            proj_n = torch.matmul(x, self.projection[n].float())  # [B, T2, D]
            proj_n = torch.nn.functional.normalize(proj_n, dim=-1)  # L2 norm on projected vector
            # codebook[n] already L2-normalized -> dot product == cosine similarity
            sim_n = torch.matmul(proj_n, self.codebook[n].float().t())  # [B, T2, V]
            targets.append(sim_n.argmax(dim=-1))  # [B, T2]
        targets = torch.stack(targets, dim=-1)  # [B, T2, N]

        target_lengths = None if lengths is None else (lengths.to(targets.device) // self.stack_size)
        return targets, target_lengths
