__all__ = ["LabelInfo", "PhoneticContext", "PhonemeStateClasses"]

from enum import Enum
from dataclasses import dataclass
import typing


class PhoneticContext(Enum):
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

    def short_name(self):
        if self == PhoneticContext.monophone:
            return "mono"
        elif self == PhoneticContext.mono_state_transition:
            return "mono_st"
        elif self == PhoneticContext.diphone:
            return "di"
        elif self == PhoneticContext.diphone_state_transition:
            return "di_st"
        elif self == PhoneticContext.triphone_symmetric:
            return "tri_sy"
        elif self == PhoneticContext.triphone_forward:
            return "tri_fw"
        elif self == PhoneticContext.triphone_backward:
            return "tri_bw"
        elif self == PhoneticContext.tri_state_transition:
            return "tri_st"
        else:
            raise AttributeError(f"unknown context enum value {self}")

    def is_monophone(self):
        return self == PhoneticContext.monophone or self == PhoneticContext.mono_state_transition

    def is_diphone(self):
        return self == PhoneticContext.diphone or self == PhoneticContext.diphone_state_transition

    def is_triphone(self):
        return (
            self == PhoneticContext.triphone_forward
            or self == PhoneticContext.triphone_backward
            or self == PhoneticContext.triphone_symmetric
            or self == PhoneticContext.tri_state_transition
        )


class PhonemeStateClasses(Enum):
    none = "none"
    boundary = "BD"
    word_end = "WE"

    def factor(self) -> int:
        if self == PhonemeStateClasses.none:
            return 1
        elif self == PhonemeStateClasses.word_end:
            return 2
        elif self == PhonemeStateClasses.boundary:
            return 4
        else:
            raise NotImplementedError("unknown phoneme state class")

    def use_boundary(self) -> bool:
        return self == PhonemeStateClasses.boundary

    def use_word_end(self) -> bool:
        return self == PhonemeStateClasses.word_end


class RasrStateTying(Enum):
    """The algorithm by which RASR calculates the labels to score."""

    monophone = "monophone-dense"
    diphone = "diphone-dense"
    triphone = "no-tying-dense"
    cart = "cart"

    def __str__(self):
        return self.value


@dataclass(eq=True, frozen=True)
class LabelInfo:
    n_contexts: int
    n_states_per_phone: int
    phoneme_state_classes: PhonemeStateClasses
    ph_emb_size: int
    st_emb_size: int
    state_tying: RasrStateTying

    add_unknown_phoneme: bool = True
    sil_id: typing.Optional[int] = None

    def get_n_of_dense_classes(self) -> int:
        n_contexts = self.n_contexts
        if not self.add_unknown_phoneme:
            n_contexts += 1
        return self.n_states_per_phone * (n_contexts**3) * self.phoneme_state_classes.factor()

    def get_n_state_classes(self) -> int:
        return self.n_states_per_phone * self.n_contexts * self.phoneme_state_classes.factor()

    @classmethod
    def default_ls(cls) -> "LabelInfo":
        return LabelInfo(
            n_contexts=42,
            n_states_per_phone=3,
            ph_emb_size=32,
            st_emb_size=128,
            sil_id=40,
            add_unknown_phoneme=True,
            phoneme_state_classes=PhonemeStateClasses.word_end,
            state_tying=RasrStateTying.triphone,
        )
