"""
Recognition-only wrapper holding a CTC ASR model *and* a standalone phoneme LM.

RETURNN builds exactly one ``Model``, so combining two separately-trained checkpoints for LM-fused
decoding means holding both as submodules and loading each from its own file via
``preload_from_files`` (``prefix="asr."`` / ``prefix="lm."``, with the forward job's
``model_checkpoint=None`` so nothing tries to load a full state dict on top).

Training never uses this class -- it exists only so the label-synchronous CTC+LM search
(``recognition.discrete_audio_ctc.label_sync_search``) can reach both models from one ``model``
argument.
"""

__all__ = ["Model"]

from typing import Any, Dict

import torch.nn as nn

from .conformer_ctc_discrete_shared_v1 import Model as CtcModel
from .transformer_decoder_lm_v1 import Model as PhonemeLm


class Model(nn.Module):
    """
    :param asr_args: net_args of the CTC model (``conformer_ctc_discrete_shared_v1.Model``).
    :param lm_args: net_args of the phoneme LM (``transformer_decoder_lm_v1.Model``).
    """

    def __init__(self, *, asr_args: Dict[str, Any], lm_args: Dict[str, Any], **_kwargs_unused):
        super().__init__()
        self.asr = CtcModel(**asr_args)
        self.lm = PhonemeLm(**lm_args)
        # the LM is frozen at recognition time; make that explicit rather than relying on eval mode
        for p in self.lm.parameters():
            p.requires_grad = False
