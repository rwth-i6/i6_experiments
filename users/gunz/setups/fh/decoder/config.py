__all__ = [
    "default_posterior_scales",
    "Float",
    "PriorConfig",
    "PriorInfo",
    "PosteriorScales",
    "SearchParameters",
]

import copy
from dataclasses import dataclass
import typing

from sisyphus import tk
from sisyphus.delayed_ops import Delayed

from ..factored import PhoneticContext


Float = typing.Union[float, tk.Variable, Delayed]


@dataclass
class PriorConfig:
    file: typing.Union[str, tk.Path, None]
    scale: Float

    def with_scale(self, scale: Float) -> "PriorConfig":
        return PriorConfig(file=self.file, scale=scale)


@dataclass
class PriorInfo:
    """
    Contains the prior XML file and scale for the left/right contexts and the
    center state.
    """

    center_state_prior: PriorConfig
    left_context_prior: typing.Optional[PriorConfig] = None
    right_context_prior: typing.Optional[PriorConfig] = None

    def with_scale(
        self,
        center: Float,
        left: typing.Optional[Float] = None,
        right: typing.Optional[Float] = None,
    ) -> "PriorInfo":
        """
        Returns a copy of the class with the given scales set.

        left/right scale must be set if the left/right priors are set.
        """

        assert self.left_context_prior is None or left is not None
        assert self.right_context_prior is None or right is not None

        left = (
            self.left_context_prior.with_scale(left)
            if self.left_context_prior is not None
            else None
        )
        right = (
            self.right_context_prior.with_scale(right)
            if self.right_context_prior is not None
            else None
        )
        return PriorInfo(
            center_state_prior=self.center_state_prior.with_scale(center),
            left_context_prior=left,
            right_context_prior=right,
        )

    @classmethod
    def from_monophone_job(cls, output_dir: typing.Union[str, tk.Path]) -> "PriorInfo":
        """
        Initializes a PriorInfo instance with scale 0.0 from the output directory of
        a previously run/captured ComputeMonophonePriorsJob.
        """

        output_dir = tk.Path(output_dir) if isinstance(output_dir, str) else output_dir
        return cls(
            center_state_prior=PriorConfig(
                file=output_dir.join_right("center-state.xml"), scale=0.0
            ),
        )

    @classmethod
    def from_diphone_job(cls, output_dir: typing.Union[str, tk.Path]) -> "PriorInfo":
        """
        Initializes a PriorInfo instance with scale 0.0 from the output directory of
        a previously run/captured ComputeDiphonePriorsJob.
        """

        output_dir = tk.Path(output_dir) if isinstance(output_dir, str) else output_dir
        return cls(
            center_state_prior=PriorConfig(
                file=output_dir.join_right("center-state.xml"), scale=0.0
            ),
            left_context_prior=PriorConfig(
                file=output_dir.join_right("left-context.xml"), scale=0.0
            ),
        )

    @classmethod
    def from_triphone_job(cls, output_dir: typing.Union[str, tk.Path]) -> "PriorInfo":
        """
        Initializes a PriorInfo instance with scale 0.0 from the output directory of
        a previously run/captured ComputeTriphoneForwardPriorsJob.
        """

        output_dir = tk.Path(output_dir) if isinstance(output_dir, str) else output_dir
        return cls(
            center_state_prior=PriorConfig(
                file=output_dir.join_right("center-state.xml"), scale=0.0
            ),
            left_context_prior=PriorConfig(
                file=output_dir.join_right("left-context.xml"), scale=0.0
            ),
            right_context_prior=PriorConfig(
                file=output_dir.join_right("right-context.xml"), scale=0.0
            ),
        )


PosteriorScales = typing.TypedDict(
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


TDP = typing.Union[Float, str]


@dataclass()
class SearchParameters:
    beam: Float
    beam_limit: int
    lm_scale: Float
    prior_info: PriorInfo
    pron_scale: Float
    # tdp_non_word: float
    tdp_scale: typing.Optional[Float]
    tdp_silence: typing.Tuple[TDP, TDP, TDP, TDP]  # loop, fwd, skip, exit
    tdp_speech: typing.Tuple[TDP, TDP, TDP, TDP]  # loop, fwd, skip, exit
    we_pruning: Float
    we_pruning_limit: int

    add_all_allophones: bool = True
    altas: typing.Optional[float] = None
    posterior_scales: typing.Optional[PosteriorScales] = None
    silence_penalties: typing.Optional[typing.Tuple[Float, Float]] = None  # loop, fwd
    state_dependent_tdps: typing.Optional[typing.Union[str, tk.Path]] = None
    transition_scales: typing.Optional[typing.Tuple[Float, Float]] = None  # loop, fwd

    def with_lm_scale(self, scale: Float) -> "SearchParameters":
        params = copy.copy(self)
        params.lm_scale = scale
        return params

    def with_prior_scale(
        self,
        center: Float,
        left: typing.Optional[Float] = None,
        right: typing.Optional[Float] = None,
    ) -> "SearchParameters":
        params = copy.copy(self)
        params.prior_info = params.prior_info.with_scale(
            center=center, left=left, right=right
        )
        return params

    def with_tdp_scale(self, scale: Float) -> "SearchParameters":
        params = copy.copy(self)
        params.tdp_scale = scale
        return params

    @classmethod
    def default_monophone(cls, *, priors: PriorInfo) -> "SearchParameters":
        return cls(
            beam=22,
            beam_limit=500_000,
            prior_info=priors.with_scale(0.2),
            lm_scale=4.0,
            tdp_scale=0.4,
            pron_scale=2.0,
            tdp_speech=(3.0, 0.0, "infinity", 0.0),
            tdp_silence=(3.0, 0.0, "infinity", 20.0),
            we_pruning=0.5,
            we_pruning_limit=10000,
        )

    @classmethod
    def default_diphone(cls, *, priors: PriorInfo) -> "SearchParameters":
        return cls(
            beam=20,
            beam_limit=500_000,
            prior_info=priors.with_scale(center=0.2, left=0.1),
            lm_scale=9.0,
            tdp_scale=0.4,
            pron_scale=2.0,
            tdp_speech=(3.0, 0.0, "infinity", 0.0),
            tdp_silence=(3.0, 0.0, "infinity", 20.0),
            we_pruning=0.5,
            we_pruning_limit=10000,
        )

    @classmethod
    def default_triphone(cls, *, priors: PriorInfo) -> "SearchParameters":
        return cls(
            beam=20,
            beam_limit=500_000,
            prior_info=priors.with_scale(center=0.2, left=0.1, right=0.1),
            lm_scale=11.0,
            tdp_scale=0.8,
            pron_scale=2.0,
            tdp_speech=(3.0, 0.0, "infinity", 0.0),
            tdp_silence=(3.0, 0.0, "infinity", 20.0),
            we_pruning=0.5,
            we_pruning_limit=10000,
        )

    @classmethod
    def default_for_ctx(
        cls, context: PhoneticContext, priors: PriorInfo
    ) -> "SearchParameters":
        if context == PhoneticContext.monophone:
            return cls.default_monophone(priors=priors)
        elif context == PhoneticContext.diphone:
            return cls.default_diphone(priors=priors)
        elif context == PhoneticContext.triphone_forward:
            return cls.default_triphone(priors=priors)
        else:
            raise NotImplementedError(f"unimplemented context {context}")
