__all__ = ["TransitionTdp", "SilenceTdp", "AmRasrConfig"]

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from sisyphus import tk

import i6_core.am as am
import i6_core.rasr as rasr


@dataclass()
class _Tdp:
    loop: Union[float, str]
    forward: Union[float, str]
    skip: Union[float, str]
    exit: Union[float, str]

    def get_tuple(
        self,
    ) -> Tuple[
        Union[float, str], Union[float, str], Union[float, str], Union[float, str]
    ]:
        return self.loop, self.forward, self.skip, self.exit


@dataclass()
class TransitionTdp(_Tdp):
    loop: Union[float, str] = 3.0
    forward: Union[float, str] = 0.0
    skip: Union[float, str] = "infinity"
    exit: Union[float, str] = 0.0


@dataclass()
class SilenceTdp(_Tdp):
    loop: Union[float, str] = 0.0
    forward: Union[float, str] = 3.0
    skip: Union[float, str] = "infinity"
    exit: Union[float, str] = 20.0


@dataclass()
class AmRasrConfig:
    state_tying: str = "monophone"
    states_per_phone: int = 3
    state_repetitions: int = 1
    across_word_model: bool = True
    early_recombination: bool = False
    tdp_scale: float = 1.0
    tdp_transition: TransitionTdp = TransitionTdp()
    tdp_silence: SilenceTdp = SilenceTdp()
    tying_type: str = "global"
    nonword_phones: str = ""
    tdp_nonword: TransitionTdp = TransitionTdp(
        0.0, 3.0, "infinity", 6.0
    )  # only used when tying_type = global-and-nonword
    state_tying_file: Optional[tk.Path] = None

    def get(self):
        am_config = am.acoustic_model_config(
            state_tying=self.state_tying,
            states_per_phone=self.states_per_phone,
            state_repetitions=self.state_repetitions,
            across_word_model=self.across_word_model,
            early_recombination=self.early_recombination,
            tdp_scale=self.tdp_scale,
            tdp_transition=self.tdp_transition,
            tdp_silence=self.tdp_silence,
            tying_type=self.tying_type,
            nonword_phones=self.nonword_phones,
            tdp_nonword=self.tdp_nonword,
        )
        if self.state_tying_file is not None:
            am_config.state_tying.file = self.state_tying_file

        return am_config
