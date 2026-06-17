"""
Lexicon-free pHMM recognition with a **count** phoneme n-gram (KenLM) as the LM,
instead of the stateful-ONNX neural LM in `rasr_phmm_lexfree_v1.py`.

Everything is identical to the lexfree neural path -- same `forward_step`, same AM
padding at the `<s>`/`</s>` indices, same `combine(no-op AM, LM)` search -- except
the LM sub-scorer is a Python KenLM label scorer registered at runtime (see
`ngram_label_scorer.register_ngram_label_scorer`). Registration happens in
`_setup_lexicon`, i.e. before the `SearchAlgorithm` is built, so RASR can resolve
the `scorer-2.type` name from its factory.
"""

from dataclasses import dataclass
from typing import Optional, Union

from sisyphus import tk

# Re-exported so the serializer can import `<module>.forward_step`.
from .rasr_forward_base import forward_step  # noqa: F401
from .rasr_phmm_lexfree_v1 import ExtraConfig, ForwardCallback as _LexfreeForwardCallback
from .ngram_label_scorer import register_ngram_label_scorer

# Must match `scorer-2.type` in build_lexiconfree_count_recognition_config.
NGRAM_SCORER_NAME = "ngram-phon"


@dataclass
class DecoderConfig:
    rasr_config_file: Union[str, tk.Path]
    lexicon: Union[str, tk.Path]
    kenlm_file: Union[str, tk.Path]
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    silence_label: str = "[SILENCE]"
    # Acoustic prior over the AM labels (see rasr_phmm_lexfree_v1.DecoderConfig); applied by the
    # inherited build_features. The count phoneme-LM lexfree path decodes the SAME CTC posteriors as
    # the lexical path, so the prior is needed here too.
    prior_file: Optional[Union[str, tk.Path]] = None
    prior_scale: float = 0.0


class ForwardCallback(_LexfreeForwardCallback):
    """Lexicon-free search whose LM sub-scorer is a registered KenLM Python label scorer."""

    def _make_config(self) -> DecoderConfig:
        return DecoderConfig(**self._config_kwargs)

    def _make_extra_config(self) -> ExtraConfig:
        return ExtraConfig(**self._extra_config_kwargs)

    def _setup_lexicon(self, lex):
        # Sets bos_index / eos_index / am_num_labels and validates bos/eos in inventory.
        super()._setup_lexicon(lex)

        from returnn.util.basic import cf

        # Register the KenLM label scorer BEFORE the SearchAlgorithm is constructed.
        register_ngram_label_scorer(
            scorer_name=NGRAM_SCORER_NAME,
            kenlm_binary_file=cf(self.config.kenlm_file),
            token_strings=self.label_inventory,
            bos_index=self.bos_index,
        )
