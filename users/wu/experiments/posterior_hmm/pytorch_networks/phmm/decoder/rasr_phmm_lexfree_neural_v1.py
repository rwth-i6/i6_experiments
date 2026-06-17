"""
Lexicon-free pHMM recognition with the **neural Transformer** phoneme LM run as a
pure-Python torch label scorer on the GPU (``neural_label_scorer``), instead of the
CPU-only stateful-ONNX scorer used by ``rasr_phmm_lexfree_v1.py``.

Everything is identical to that ONNX path -- same ``forward_step``, same AM padding
at the ``<s>``/``</s>`` indices, same cascade ``label-scorer-1=no-op AM`` +
``label-scorer-2=<python LM>`` search -- except the LM sub-scorer is the torch LM
itself, registered at runtime (see ``neural_label_scorer.register_neural_label_scorer``).
Registration happens in ``_setup_lexicon``, i.e. before the ``SearchAlgorithm`` is
built, so RASR can resolve the ``label-scorer-2.type`` name from its factory.

This mirrors the count-LM decoder ``rasr_phmm_lexfree_ngram_v1.py`` (which proves the
Python-label-scorer plumbing end-to-end); only the scorer internals differ
(kenlm.State -> torch KV cache, ``full_scores`` -> ``_single_step`` logits on GPU).
Run the forward job with a GPU (``use_gpu=True``) so both the AM forward and this LM
scorer execute on the device.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Union

from sisyphus import tk

# Re-exported so the serializer can import ``<module>.forward_step``.
from .rasr_forward_base import forward_step  # noqa: F401
from .rasr_phmm_lexfree_v1 import ExtraConfig, ForwardCallback as _LexfreeForwardCallback
from .neural_label_scorer import register_neural_label_scorer

# Must match ``label-scorer-2.type`` in the RASR config (build_lexiconfree_neural_python_recognition_config).
NEURAL_SCORER_NAME = "neural-phon"


@dataclass
class DecoderConfig:
    rasr_config_file: Union[str, tk.Path]
    lexicon: Union[str, tk.Path]
    lm_checkpoint: Union[str, tk.Path]
    lm_net_args: Dict[str, Any] = field(default_factory=dict)
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    silence_label: str = "[SILENCE]"


class ForwardCallback(_LexfreeForwardCallback):
    """Lexicon-free search whose LM sub-scorer is a registered torch-LM Python label scorer."""

    def _make_config(self) -> DecoderConfig:
        return DecoderConfig(**self._config_kwargs)

    def _make_extra_config(self) -> ExtraConfig:
        return ExtraConfig(**self._extra_config_kwargs)

    def _setup_lexicon(self, lex):
        # Sets bos_index / eos_index / am_num_labels and validates bos/eos in inventory.
        super()._setup_lexicon(lex)

        # Register the torch LM label scorer BEFORE the SearchAlgorithm is constructed.
        register_neural_label_scorer(
            scorer_name=NEURAL_SCORER_NAME,
            lm_checkpoint=self.config.lm_checkpoint,
            lm_net_args=self.config.lm_net_args,
            bos_index=self.bos_index,
            num_labels=self.num_labels,
        )
