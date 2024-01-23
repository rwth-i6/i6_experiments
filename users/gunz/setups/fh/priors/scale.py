__all__ = ["scale_priors", "ScalePriorsJob"]

import dataclasses
import numpy as np
import typing

from sisyphus import Job, Path, Task

from ..decoder.config import PriorConfig, PriorInfo
from .util import read_prior_xml, write_prior_xml

Indices = typing.Union[typing.List[int], typing.List[typing.Tuple[int, int]], typing.List[typing.Tuple[int, int, int]]]


def scale_priors(
    p_info: PriorInfo,
    scale_c: float = 1.0,
    scale_l: float = 1.0,
    scale_r: float = 1.0,
) -> PriorInfo:
    def scale(cfg: typing.Optional[PriorConfig], scale: float) -> typing.Optional[PriorConfig]:
        if cfg is None:
            return None

        job = ScalePriorsJob(cfg.file, scale)
        return dataclasses.replace(cfg, file=job.out_priors)

    return dataclasses.replace(
        p_info,
        left_context_prior=scale(p_info.left_context_prior, scale_l),
        center_state_prior=scale(p_info.center_state_prior, scale_c),
        right_context_prior=scale(p_info.right_context_prior, scale_r),
    )


class ScalePriorsJob(Job):
    """
    Scales computed priors by a constant value.

    Computes `np.array(priors) * scale`.
    """

    def __init__(self, prior_xml: Path, scale: float):
        self.prior_xml = prior_xml
        self.scale = scale

        self.out_priors = self.output_path("priors.xml")

    def tasks(self) -> typing.Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        parse_result = read_prior_xml(self.prior_xml)

        priors = np.exp(parse_result.priors_log).reshape(parse_result.shape) * self.scale

        total_weight = priors.flatten().sum()
        print(f"total prior weight: {total_weight}")

        write_prior_xml(np.log(priors), self.out_priors)
