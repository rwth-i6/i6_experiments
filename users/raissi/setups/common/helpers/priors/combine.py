__all__ = ["combine_priors_across_hmm_states"]

import dataclasses
import numpy as np
from typing import Iterator

from sisyphus import Job, Path, Task

from i6_experiments.users.raissi.setups.common.data.factored_label import LabelInfo
from i6_experiments.users.raissi.setups.common.decoder.config import PriorInfo
from i6_experiments.users.raissi.setups.common.helpers.priors.util import read_prior_xml, write_prior_xml


def combine_priors_across_hmm_states(
    prior_info: PriorInfo, label_info_in: LabelInfo, label_info_out: LabelInfo
) -> PriorInfo:
    assert label_info_in.n_states_per_phone >= label_info_out.n_states_per_phone, "cannot spread probabilities"

    if label_info_in.n_states_per_phone == label_info_out.n_states_per_phone:
        return prior_info

    center_states = CombinePriorsAcrossHmmStatesJob(prior_info.center_state_prior.file, label_info_in, label_info_out)
    center_priors = dataclasses.replace(prior_info.center_state_prior, file=center_states.out_priors)

    if prior_info.right_context_prior is not None:
        right_contexts = CombinePriorsAcrossHmmStatesJob(
            prior_info.right_context_prior.file, label_info_in, label_info_out
        )
        right_priors = dataclasses.replace(prior_info.right_context_prior, file=right_contexts.out_priors)
    else:
        right_priors = None

    return dataclasses.replace(prior_info, center_state_prior=center_priors, right_context_prior=right_priors)


class CombinePriorsAcrossHmmStatesJob(Job):
    def __init__(self, prior_xml: Path, label_info_in: LabelInfo, label_info_out: LabelInfo):
        assert label_info_in.n_contexts == label_info_out.n_contexts
        assert label_info_in.phoneme_state_classes == label_info_out.phoneme_state_classes
        assert label_info_out.n_states_per_phone == 1

        self.label_info_in = label_info_in
        self.label_info_out = label_info_out
        self.prior_xml = prior_xml

        self.out_priors = self.output_path("priors.xml")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        parsed = read_prior_xml(self.prior_xml)

        priors = np.exp(parsed.priors_log)
        shaped = np.reshape(
            priors,
            (
                self.label_info_in.n_contexts,
                self.label_info_in.n_contexts if len(parsed.shape) > 1 else 1,
                self.label_info_in.n_states_per_phone,
                self.label_info_in.phoneme_state_classes.factor(),
                -1,  # this gives us support for the right context priors
            ),
        )
        sum_across = np.sum(shaped, axis=2)
        log_priors = np.log(sum_across)

        if len(parsed.shape) == 1:
            target_shape = (-1,)
        else:
            target_shape = (-1, parsed.shape[1])

        write_prior_xml(np.reshape(log_priors, target_shape), self.out_priors)
