__all__ = ["Tdp", "SpeechTdp", "SilenceTdp", "NonSpeechTdp", "AmRasrConfig"]

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union


from sisyphus import tk

import i6_core.rasr as rasr


class StateTying(Enum):
    MONOPHONE = 1
    CART = 2
    DENSE = 3

    def __str__(self):
        if self.value == 1:
            return "monophone"
        elif self.value == 2:
            return "cart"
        elif self.value == 3:
            return "dense"
        else:
            raise NotImplemented


@dataclass()
class Tdp:
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

    def __str__(self):
        return f"loop{self.loop}_forward{self.forward}_skip{self.skip}_exit{self.exit}"


@dataclass()
class SpeechTdp(Tdp):
    loop: Union[float, str] = field(default=3.0)
    forward: Union[float, str] = field(default=0.0)
    skip: Union[float, str] = field(default="infinity")
    exit: Union[float, str] = field(default=0.0)


@dataclass()
class SilenceTdp(Tdp):
    loop: Union[float, str] = field(default=0.0)
    forward: Union[float, str] = field(default=3.0)
    skip: Union[float, str] = field(default="infinity")
    exit: Union[float, str] = field(default=20.0)


@dataclass()
class NonSpeechTdp(Tdp):
    loop: Union[float, str] = field(default=0.0)
    forward: Union[float, str] = field(default=3.0)
    skip: Union[float, str] = field(default="infinity")
    exit: Union[float, str] = field(default=6.0)


def acoustic_model_config(
    state_tying: StateTying = StateTying.MONOPHONE,
    state_tying_file: Optional[tk.Path] = None,
    states_per_phone: int = 3,
    state_repetitions: int = 1,
    across_word_model: bool = True,
    early_recombination: bool = False,
    tying_type: str = "global",
    nonword_phones: Optional[List[str]] = None,
    allophones: Optional[tk.Path] = None,
) -> rasr.RasrConfig:
    """

    :param state_tying: monophone, cart, dense-tying, ...
    :param state_tying_file:
    :param states_per_phone:
    :param state_repetitions:
    :param across_word_model:
    :param early_recombination:
    :param tying_type: global, global-and-nonword, ...
    :param nonword_phones:
    :param allophones:
    :return: acoustic model rasr config
    """
    config = rasr.RasrConfig()

    config.state_tying.type = state_tying
    config.state_tying.file = state_tying_file

    config.allophones.add_all = False
    config.allophones.add_from_lexicon = True if allophones is None else False
    if allophones is not None:
        config.allophones.add_from_file = allophones

    config.hmm.states_per_phone = states_per_phone
    config.hmm.state_repetitions = state_repetitions
    config.hmm.across_word_model = across_word_model
    config.hmm.early_recombination = early_recombination

    config.tdp["entry-m1"].loop = "infinity"
    config.tdp["entry-m2"].loop = "infinity"

    if tying_type == "global-and-nonword":
        config.tdp.tying_type = "global-and-nonword"
        config.tdp.nonword_phones = ",".join(nonword_phones)

    return config


@dataclass()
class AmRasrConfig:
    state_tying: StateTying = StateTying.MONOPHONE
    state_tying_file: Optional[tk.Path] = None
    states_per_phone: int = 3
    state_repetitions: int = 1
    across_word_model: bool = True
    early_recombination: bool = False
    tying_type: str = "global"
    nonword_phones: List[str] = ""

    def get(self):
        am_config = acoustic_model_config(
            state_tying=self.state_tying,
            state_tying_file=self.state_tying_file,
            states_per_phone=self.states_per_phone,
            state_repetitions=self.state_repetitions,
            across_word_model=self.across_word_model,
            early_recombination=self.early_recombination,
            tying_type=self.tying_type,
            nonword_phones=self.nonword_phones,
        )

        return am_config
