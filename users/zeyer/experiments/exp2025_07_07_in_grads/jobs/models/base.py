"""
Base interface
"""

from __future__ import annotations
from typing import Optional, Union, Any, List, Tuple
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
    cls = _get_cls(cls_name)
    return cls(**opts)


class BaseModelInterface(torch.nn.Module):
    """
    Base interface for all models.
    """

    assistant_end_token_idx: Optional[int] = None

    def forward(
        self,
        *,
        raw_inputs: Union[np.ndarray, torch.Tensor, List[List[str]]],
        raw_inputs_sample_rate: Optional[int] = None,
        raw_input_seq_lens: torch.Tensor,
        raw_targets: List[List[str]],
        raw_target_seq_lens: torch.Tensor,
        omitted_prev_context: Optional[torch.Tensor] = None,
    ) -> ForwardOutput:
        """
        Process and (maybe partially) forward.
        Then :func:`score` is supposed to be called for each target frame.

        :param raw_inputs: Input seqs. Shape [B,T_in_raw,...], e.g. audio raw samples, or words.
        :param raw_inputs_sample_rate: Sample rate of the input audio, if applicable.
        :param raw_input_seq_lens: Length of each input sequence.
        :param raw_targets: Target seqs. List of words.
        :param raw_target_seq_lens: Length of each target sequence.
        :param omitted_prev_context: Specifies if there was any prev context omitted
            (e.g. if there was a prev chunk, which had at least some words).
            Shape [B]. Num raw targets.
        :return: Forward up to a common point. The :func:`score` could still do further computation on top,
            but this would be separate for every target frame.
            Might be useful to save memory.
            E.g. the LLM head (final linear transformation to logits) could be
        """
        raise NotImplementedError("Forward method must be implemented by the subclass.")

    def log_probs(
        self, *, forward_output: ForwardOutput, start: Union[int, torch.Tensor], end: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        Calculate the score for a given target frame index.

        :param forward_output:
        :param start: start position in the target sequence, can be scalar or tensor of shape [B]
        :param end: end position (excluding) in the target sequence, can be scalar or tensor of shape [B]
            For start/end, you can use this:
            ``start, end = forward_output.target_start_end[:, raw_target_frame_index].unbind(1)  # [B], [B]``.
        :return: Log probs for the specified target frames, shape [B,T,V], where T = end-start.
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
    input_seq_lens: torch.Tensor  # [B]: Length of each input sequence (values in [0..T_in])
    input_slice_start_end: Optional[Tuple[torch.Tensor, torch.Tensor]]  # (start, end) of input slices, each [B]
    # If input_slice_start_end is set, we will only consider this slice (for saliency etc),
    # And this results in shorter T_in', i.e. inputs' shape [B, T_in', F_in].
    # See :func:`.utils.apply_input_slice` to apply this slice.
    input_raw_start_end: torch.Tensor  # [B, T_in', 2] -> T_in_raw: (start, end) of raw input frames
    # (both start and end are inclusive)

    targets: torch.Tensor  # [B, T_out] -> class indices
    target_seq_lens: torch.Tensor  # [B]: Length of each target sequence
    target_start_end: torch.Tensor  # [B, T_out_raw, 2] -> T_out: (start, end) of raw target frames
    # (both start and end are inclusive)

    outputs: Any  # any nested structure

    def get_inputs_seq_lens_sliced(self) -> torch.Tensor:
        if self.input_slice_start_end is None:
            return self.input_seq_lens
        start, end = self.input_slice_start_end
        return end - start


_classes = {}


def _get_cls(cls_name: str) -> type:
    """
    Get class by name.
    """
    if "." in cls_name:
        pkg_name, base_cls_name = cls_name.rsplit(".", 1)
        mod = importlib.import_module(pkg_name, package=__package__)
        cls = getattr(mod, base_cls_name)
        if not issubclass(cls, BaseModelInterface):
            raise TypeError(f"Class {cls_name} must inherit from BaseModelInterface.")
        return cls

    if not _classes:
        from . import phi4mm

        for mod in [phi4mm]:
            for name, cls in vars(mod).items():
                if isinstance(cls, type) and issubclass(cls, BaseModelInterface):
                    _classes[name] = cls

        assert _classes

    if cls_name not in _classes:
        raise ValueError(f"Class {cls_name} not found in available classes: {list(_classes.keys())}")
    return _classes[cls_name]
