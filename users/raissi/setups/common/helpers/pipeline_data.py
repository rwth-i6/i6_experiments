__all__ = ["ContextEnum", "ContextMapper", "PipelineStages", "LabelInfo"]

from enum import Enum


class ContextEnum(Enum):
    """
    These are the implemented models. The string value is the one used in the feature scorer of rasr, except monophone
    """

    monophone = "monophone"
    mono_state_transition = "monophone-delta"
    diphone = "diphone"
    diphone_state_transition = "diphone-delta"
    triphone_symmetric = "triphone-symmetric"
    triphone_forward = "triphone-forward"
    triphone_backward = "triphone-backward"
    tri_state_transition = "triphone-delta"


class ContextMapper:
    def __init__(self):
        self.contexts = {
            1: "monophone",
            2: "diphone",
            3: "triphone-symmetric",
            4: "triphone-forward",
            5: "triphone-backward",
            6: "triphone-delta",
            7: "monophone-delta",
            8: "diphone-delta",
        }

    def get_enum(self, contextTypeId):
        return self.contexts[contextTypeId]


class LabelInfo:
    def __init__(
        self,
        n_states_per_phone,
        n_phonemes,
        ph_emb_size,
        st_emb_size,
        state_tying,
        sil_id=None,
        use_word_end_class=True,
        use_boundary_classes=False,
    ):
        self.n_states_per_phone = n_states_per_phone
        self.n_contexts = n_phonemes
        self.n_state_classes = n_states_per_phone * n_phonemes
        self.sil_id = sil_id
        self.ph_emb_size = ph_emb_size
        self.st_emb_size = st_emb_size
        self.state_tying = state_tying
        self.use_word_end_class = use_word_end_class
        self.use_boundary_classes = use_boundary_classes


class PipelineStages:
    def __init__(self, alignment_keys):
        self.names = dict(
            zip(alignment_keys, [self._get_context_dict(k) for k in alignment_keys])
        )

    def _get_context_dict(self, label):
        return {
            "mono": f"mono-from-{label}",
            "mono-delta": f"mono-delta-from-{label}",
            "di": f"di-from-{label}",
            "di-delta": f"di-delta-from-{label}",
            "tri": f"tri-from-{label}",
            "tri-delta": f"tridelta-from-{label}",
        }

    def get_name(self, alignment_key, context_type):
        return self.names[alignment_key][context_type]
