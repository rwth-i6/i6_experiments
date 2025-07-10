"""
Base interface
"""

from __future__ import annotations
from typing import Optional, Union, Any, List
import torch
import numpy as np
import importlib
from dataclasses import dataclass


def make_model(**opts) -> BaseModelInterface:
    """
    Make model wrapper
    """
    opts = opts.copy()
    cls_name = opts.pop("type")
    pkg_name, base_cls_name = cls_name.rsplit(".", 1)
    mod = importlib.import_module(pkg_name, package=__package__)
    cls = getattr(mod, base_cls_name)
    if not issubclass(cls, BaseModelInterface):
        raise TypeError(f"Class {cls_name} must inherit from BaseModelInterface.")
    return cls(**opts)


class BaseModelInterface(torch.nn.Module):
    """
    Base interface for all models.
    """

    def forward(
        self,
        *,
        raw_inputs: Union[np.ndarray, torch.Tensor, List[List[str]]],
        raw_inputs_sample_rate: Optional[int] = None,
        raw_input_seq_lens: torch.Tensor,
        raw_targets: List[List[str]],
        raw_target_seq_lens: torch.Tensor,
    ) -> ForwardOutput:
        """
        Process and (maybe partially) forward.
        Then :func:`score` is supposed to be called for each target frame.

        :param raw_inputs: Input seqs. Shape [B,T_in_raw,...], e.g. audio raw samples, or words.
        :param raw_inputs_sample_rate: Sample rate of the input audio, if applicable.
        :param raw_input_seq_lens: Length of each input sequence.
        :param raw_targets: Target seqs. List of words.
        :param raw_target_seq_lens: Length of each target sequence.
        :return: Forward up to a common point. The :func:`score` could still do further computation on top,
            but this would be separate for every target frame.
            Might be useful to save memory.
            E.g. the LLM head (final linear transformation to logits) could be
        """
        raise NotImplementedError("Forward method must be implemented by the subclass.")

    def score(self, *, forward_output: ForwardOutput, raw_target_frame_index: int) -> torch.Tensor:
        """
        Calculate the score for a given target frame index.

        :param forward_output:
        :param raw_target_frame_index: in [0..T_out_raw-1] range, index of the target frame to score.
        :return: Score tensor for the specified target frame.
        """
        raise NotImplementedError("Score method must be implemented by the subclass.")


@dataclass
class ForwardOutput:
    """
    Data class for preprocessed input/target sequence pairs.

    `inputs` here is where we will calculate grads wrt,
    for each target in `targets`.
    """

    inputs: torch.Tensor  # [B, T_in, F_in]. e.g. audio embeddings. grads will be calculated w.r.t. this.
    input_seq_lens: torch.Tensor  # [B]: Length of each input sequence
    input_raw_start_end: torch.Tensor  # [B, T_in, 2] -> T_in_raw: (start, end) of raw input frames
    # (both start and end are inclusive)

    targets: torch.Tensor  # [B, T_out] -> class indices
    target_seq_lens: torch.Tensor  # [B]: Length of each target sequence
    target_start_end: torch.Tensor  # [B, T_out_raw, 2] -> T_out: (start, end) of raw target frames
    # (both start and end are inclusive)

    outputs: Any  # any nested structure
