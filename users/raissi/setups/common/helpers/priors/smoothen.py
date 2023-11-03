__all__ = ["smoothen_priors", "SmoothenPriorsJob"]

import dataclasses
import numpy as np
import typing

from sisyphus import Job, Path, Task

from i6_experiments.users.raissi.setups.common.decoder.config import PriorInfo, PriorConfig
from i6_experiments.users.raissi.setups.common.helpers.priors.util import read_prior_xml, write_prior_xml

Indices = typing.Union[typing.List[int], typing.List[typing.Tuple[int, int]], typing.List[typing.Tuple[int, int, int]]]


def smoothen_priors(
    p_info: PriorInfo,
    zero_weight: float = 1e-8,
    combine_indices_l: typing.Optional[Indices] = None,
    combine_indices_c: typing.Optional[Indices] = None,
    combine_indices_r: typing.Optional[Indices] = None,
) -> PriorInfo:
    def smoothen(
        cfg: typing.Optional[PriorConfig], zero_w: float, combine_idx: typing.Optional[Indices]
    ) -> typing.Optional[PriorConfig]:
        if cfg is None:
            return None

        job = SmoothenPriorsJob(cfg.file, zero_w, combine_idx)
        return dataclasses.replace(cfg, file=job.out_priors)

    return dataclasses.replace(
        p_info,
        left_context_prior=smoothen(p_info.left_context_prior, zero_weight, combine_indices_l),
        center_state_prior=smoothen(p_info.center_state_prior, zero_weight, combine_indices_c),
        right_context_prior=smoothen(p_info.right_context_prior, zero_weight, combine_indices_r),
    )


class SmoothenPriorsJob(Job):
    """
    Smoothens computed priors by setting the zero priors to a (small) base value and potentially
    combines the weights for certain indices of priors.
    """

    def __init__(
        self,
        prior_xml: Path,
        zero_weight: float = 1e-8,
        combine_indices: typing.Optional[Indices] = None,
    ):
        self.combine_indices = combine_indices
        self.prior_xml = prior_xml
        self.zero_weight = zero_weight

        self.out_priors = self.output_path("priors.xml")

    def tasks(self) -> typing.Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        parse_result = read_prior_xml(self.prior_xml)

        priors = np.exp(parse_result.priors_log).reshape(parse_result.shape)

        zero_priors = priors == 0
        print(f"smoothing {zero_priors.size} priors to {self.zero_weight:.2E}")
        priors[zero_priors] = self.zero_weight

        if self.combine_indices is not None:
            to_combine = np.array([priors[idx] for idx in self.combine_indices])
            combined = np.sum(to_combine, axis=0)

            for idx in self.combine_indices:
                priors[idx] = combined

        total_weight = priors.flatten().sum()
        print(f"total prior weight: {total_weight}")

        write_prior_xml(np.log(priors), self.out_priors)
