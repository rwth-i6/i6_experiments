__all__ = ["default_posterior_scales", "Float", "PriorConfig", "PriorInfo", "PosteriorScales", "SearchParameters"]

import dataclasses
from dataclasses import dataclass
from typing import Optional, Union, Tuple, TypedDict

from sisyphus import tk

from i6_experiments.users.raissi.setups.common.data.typings import (
    Int,
    Float,
    TDP,
)

from i6_experiments.users.raissi.setups.common.data.factored_label import (
    PhoneticContext,
)


@dataclass(eq=True, frozen=True)
class PriorConfig:
    file: Union[str, tk.Path, None]
    scale: Float

    def with_scale(self, scale: Float) -> "PriorConfig":
        return PriorConfig(file=self.file, scale=scale)


@dataclass(eq=True, frozen=True)
class PriorInfo:
    """
    Contains the prior XML file and scale for the left/right contexts and the
    center state.
    """

    center_state_prior: Optional[PriorConfig] = None
    left_context_prior: Optional[PriorConfig] = None
    right_context_prior: Optional[PriorConfig] = None
    diphone_prior: Optional[PriorConfig] = None

    def with_scale(
        self,
        center: Optional[Float] = None,
        left: Optional[Float] = None,
        right: Optional[Float] = None,
        diphone: Optional[Float] = None,
    ) -> "PriorInfo":
        """
        Returns a copy of the class with the given scales set.

        left/right/diphone scale must be set if the left/right priors are set.
        """
        assert self.center_state_prior is None or center is not None
        assert self.left_context_prior is None or left is not None
        assert self.right_context_prior is None or right is not None
        assert self.diphone_prior is None or diphone is not None

        center = self.center_state_prior.with_scale(center) if self.center_state_prior is not None else None
        left = self.left_context_prior.with_scale(left) if self.left_context_prior is not None else None
        right = self.right_context_prior.with_scale(right) if self.right_context_prior is not None else None
        diphone = self.diphone_prior.with_scale(diphone) if self.diphone_prior is not None else None
        return dataclasses.replace(
            self,
            center_state_prior=center,
            left_context_prior=left,
            right_context_prior=right,
            diphone_prior=diphone,
        )

    @classmethod
    def from_monophone_job(cls, output_dir: Union[str, tk.Path]) -> "PriorInfo":
        """
        Initializes a PriorInfo instance with scale 0.0 from the output directory of
        a previously run/captured ComputeMonophonePriorsJob.
        """

        output_dir = tk.Path(output_dir) if isinstance(output_dir, str) else output_dir
        return cls(
            center_state_prior=PriorConfig(file=output_dir.join_right("center-state.xml"), scale=0.0),
        )

    @classmethod
    def from_diphone_job(cls, output_dir: Union[str, tk.Path]) -> "PriorInfo":
        """
        Initializes a PriorInfo instance with scale 0.0 from the output directory of
        a previously run/captured ComputeDiphonePriorsJob.
        """

        output_dir = tk.Path(output_dir) if isinstance(output_dir, str) else output_dir
        return cls(
            center_state_prior=PriorConfig(file=output_dir.join_right("center-state.xml"), scale=0.0),
            left_context_prior=PriorConfig(file=output_dir.join_right("left-context.xml"), scale=0.0),
        )

    @classmethod
    def from_triphone_job(cls, output_dir: Union[str, tk.Path]) -> "PriorInfo":
        """
        Initializes a PriorInfo instance with scale 0.0 from the output directory of
        a previously run/captured ComputeTriphoneForwardPriorsJob.
        """

        output_dir = tk.Path(output_dir) if isinstance(output_dir, str) else output_dir
        return cls(
            center_state_prior=PriorConfig(file=output_dir.join_right("center-state.xml"), scale=0.0),
            left_context_prior=PriorConfig(file=output_dir.join_right("left-context.xml"), scale=0.0),
            right_context_prior=PriorConfig(file=output_dir.join_right("right-context.xml"), scale=0.0),
        )

    def from_diiphone_job(cls, output_dir: Union[str, tk.Path]) -> "PriorInfo":
        """
        Initializes a PriorInfo instance with scale 0.0 from the output directory of
        a previously run/captured ComputeTriphoneForwardPriorsJob.
        """

        output_dir = tk.Path(output_dir) if isinstance(output_dir, str) else output_dir
        return cls(
            diphone_prior=PriorConfig(file=output_dir.join_right("center-state.xml"), scale=0.0),
        )


PosteriorScales = TypedDict(
    "PosteriorScales",
    {
        "left-context-scale": Float,
        "right-context-scale": Float,
        "center-state-scale": Float,
    },
)


def default_posterior_scales() -> PosteriorScales:
    return {
        "left-context-scale": 1.0,
        "right-context-scale": 1.0,
        "center-state-scale": 1.0,
    }


@dataclass(eq=True, frozen=True)
class SearchParameters:
    beam: Float
    beam_limit: Int
    lm_scale: Float
    non_word_phonemes: str
    prior_info: PriorInfo
    pron_scale: Float
    tdp_scale: Optional[Float]
    tdp_silence: Tuple[TDP, TDP, TDP, TDP]  # loop, fwd, skip, exit
    tdp_speech: Tuple[TDP, TDP, TDP, TDP]  # loop, fwd, skip, exit
    tdp_non_word: Tuple[TDP, TDP, TDP, TDP]  # loop, fwd, skip, exit
    we_pruning: Float
    we_pruning_limit: Int

    add_all_allophones: bool = True
    altas: Optional[float] = None
    posterior_scales: Optional[PosteriorScales] = None
    silence_penalties: Optional[Tuple[Float, Float]] = None  # loop, fwd
    state_dependent_tdps: Optional[Union[str, tk.Path]] = None
    transition_scales: Optional[Tuple[Float, Float]] = None  # loop, fwd

    def with_altas(self, altas: Float) -> "SearchParameters":
        return dataclasses.replace(self, altas=altas)

    def with_beam_limit(self, beam_limit: Int) -> "SearchParameters":
        return dataclasses.replace(self, beam_limit=beam_limit)

    def with_beam_size(self, beam: Float) -> "SearchParameters":
        return dataclasses.replace(self, beam=beam)

    def with_lm_scale(self, scale: Float) -> "SearchParameters":
        return dataclasses.replace(self, lm_scale=scale)

    def with_prior_scale(
        self,
        center: Optional[Float] = None,
        left: Optional[Float] = None,
        right: Optional[Float] = None,
        diphone: Optional[Float] = None,
    ) -> "SearchParameters":
        return dataclasses.replace(
            self, prior_info=self.prior_info.with_scale(center=center, left=left, right=right, diphone=diphone)
        )

    def with_pron_scale(self, pron_scale: Float) -> "SearchParameters":
        return dataclasses.replace(self, pron_scale=pron_scale)

    def with_tdp_non_word(self, tdp: Tuple[TDP, TDP, TDP, TDP]) -> "SearchParameters":
        return dataclasses.replace(self, tdp_non_word=tdp)

    def with_tdp_scale(self, scale: Float) -> "SearchParameters":
        return dataclasses.replace(self, tdp_scale=scale)

    def with_tdp_silence(self, tdp: Tuple[TDP, TDP, TDP, TDP]) -> "SearchParameters":
        return dataclasses.replace(self, tdp_silence=tdp)

    def with_tdp_speech(self, tdp: Tuple[TDP, TDP, TDP, TDP]) -> "SearchParameters":
        return dataclasses.replace(self, tdp_speech=tdp)

    @classmethod
    def default_monophone(cls, *, priors: PriorInfo) -> "SearchParameters":
        return cls(
            beam=22,
            beam_limit=500_000,
            lm_scale=4.0,
            tdp_scale=0.4,
            pron_scale=2.0,
            prior_info=priors.with_scale(0.2),
            tdp_speech=(3.0, 0.0, "infinity", 0.0),
            tdp_silence=(0.0, 3.0, "infinity", 20.0),
            tdp_non_word=(0.0, 3.0, "infinity", 20.0),
            non_word_phonemes="[UNKNOWN]",
            we_pruning=0.5,
            we_pruning_limit=10000,
        )

    @classmethod
    def default_diphone(cls, *, priors: PriorInfo) -> "SearchParameters":
        return cls(
            beam=20,
            beam_limit=500_000,
            lm_scale=9.0,
            tdp_scale=0.4,
            pron_scale=2.0,
            prior_info=priors.with_scale(center=0.2, left=0.1),
            tdp_speech=(3.0, 0.0, "infinity", 0.0),
            tdp_silence=(0.0, 3.0, "infinity", 20.0),
            tdp_non_word=(0.0, 3.0, "infinity", 20.0),
            non_word_phonemes="[UNKNOWN]",
            we_pruning=0.5,
            we_pruning_limit=10000,
        )

    @classmethod
    def default_triphone(cls, *, priors: PriorInfo) -> "SearchParameters":
        return cls(
            beam=20,
            beam_limit=500_000,
            lm_scale=11.0,
            tdp_scale=0.6,
            pron_scale=2.0,
            prior_info=priors.with_scale(center=0.2, left=0.1, right=0.1),
            tdp_speech=(3.0, 0.0, "infinity", 0.0),
            tdp_silence=(0.0, 3.0, "infinity", 20.0),
            tdp_non_word=(0.0, 3.0, "infinity", 20.0),
            non_word_phonemes="[UNKNOWN]",
            we_pruning=0.5,
            we_pruning_limit=10000,
        )

    @classmethod
    def default_joint_diphone(cls, *, priors: PriorInfo) -> "SearchParameters":
        return cls(
            beam=20,
            beam_limit=500_000,
            lm_scale=9.0,
            tdp_scale=0.4,
            pron_scale=2.0,
            prior_info=priors.with_scale(diphone=0.4),
            tdp_speech=(3.0, 0.0, "infinity", 0.0),
            tdp_silence=(0.0, 3.0, "infinity", 20.0),
            tdp_non_word=(0.0, 3.0, "infinity", 20.0),
            non_word_phonemes="[UNKNOWN]",
            we_pruning=0.5,
            we_pruning_limit=10000,
        )

    @classmethod
    def default_cart(cls, *, priors: PriorInfo) -> "SearchParameters":
        return dataclasses.replace(
            cls.default_triphone(priors=priors.with_scale(center=0.3)),
            beam=16,
            beam_limit=100_000,
            lm_scale=12,
        )

    @classmethod
    def default_for_ctx(cls, context: PhoneticContext, priors: PriorInfo) -> "SearchParameters":
        if context == PhoneticContext.monophone:
            return cls.default_monophone(priors=priors)
        elif context == PhoneticContext.diphone:
            return cls.default_diphone(priors=priors)
        elif context == PhoneticContext.triphone_forward:
            return cls.default_triphone(priors=priors)
        elif context == PhoneticContext.joint_diphone:
            return cls.default_joint_diphone(priors=priors)

        else:
            raise NotImplementedError(f"unimplemented context {context}")