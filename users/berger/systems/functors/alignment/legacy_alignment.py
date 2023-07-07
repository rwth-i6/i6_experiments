import copy
import itertools
from typing import Dict, List

from i6_core import mm, rasr, returnn
from sisyphus import tk

from ... import dataclasses
from ... import types
from ..base import AlignmentFunctor
from ..rasr_base import RasrFunctor


class LegacyAlignmentFunctor(
    AlignmentFunctor[returnn.ReturnnTrainingJob, returnn.ReturnnConfig],
    RasrFunctor,
):
    def __call__(
        self,
        train_job: dataclasses.NamedTrainJob[returnn.ReturnnTrainingJob],
        prior_config: returnn.ReturnnConfig,
        align_config: returnn.ReturnnConfig,
        align_corpus: dataclasses.NamedCorpusInfo,
        num_inputs: int,
        num_classes: int,
        epochs: List[types.EpochType] = [],
        prior_scales: List[float] = [0],
        prior_args: Dict = {},
        feature_type: dataclasses.FeatureType = dataclasses.FeatureType.SAMPLES,
        flow_args: Dict = {},
        **kwargs,
    ) -> None:
        crp = copy.deepcopy(align_corpus.corpus_info.crp)

        acoustic_mixture_path = mm.CreateDummyMixturesJob(num_classes, num_inputs).out_mixtures

        base_feature_flow = self._make_base_feature_flow(
            align_corpus.corpus_info, feature_type=feature_type, **flow_args
        )

        for prior_scale, epoch in itertools.product(prior_scales, epochs):
            tf_graph = self._make_tf_graph(
                train_job=train_job.job,
                returnn_config=align_config,
                epoch=epoch,
            )

            checkpoint = self._get_checkpoint(train_job.job, epoch)
            prior_file = self._get_prior_file(
                prior_config=prior_config,
                checkpoint=checkpoint,
                **prior_args,
            )

            feature_scorer = rasr.PrecomputedHybridFeatureScorer(
                prior_mixtures=acoustic_mixture_path,
                priori_scale=prior_scale,
                prior_file=prior_file,
            )

            feature_flow = self._make_tf_feature_flow(
                base_feature_flow,
                tf_graph,
                checkpoint,
            )

            align = mm.AlignmentJob(
                crp=crp,
                feature_flow=feature_flow,
                feature_scorer=feature_scorer,
                **kwargs,
            )

            exp_full = f"align_e-{self._get_epoch_string(epoch)}_prior-{prior_scale:02.2f}"
            path = f"nn_recog/{align_corpus.name}/{train_job.name}/{exp_full}"

            align.set_vis_name(f"Alignment {path}")
            align.add_alias(path)

            tk.register_output(
                f"{path}.alignment.cache.bundle",
                align.out_alignment_bundle,
            )
