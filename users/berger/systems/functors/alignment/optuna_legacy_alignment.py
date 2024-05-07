import copy
from typing import Dict

from i6_core import mm, rasr
from i6_core.lexicon import DumpStateTyingJob, StoreAllophonesJob
from i6_experiments.users.berger.recipe import returnn
from sisyphus import tk

from ... import dataclasses
from ... import types
from ..base import AlignmentFunctor
from ..optuna_rasr_base import OptunaRasrFunctor


class OptunaLegacyAlignmentFunctor(
    AlignmentFunctor[returnn.OptunaReturnnTrainingJob, returnn.OptunaReturnnConfig],
    OptunaRasrFunctor,
):
    def __call__(
        self,
        train_job: dataclasses.NamedTrainJob[returnn.OptunaReturnnTrainingJob],
        prior_config: returnn.OptunaReturnnConfig,
        align_config: returnn.OptunaReturnnConfig,
        align_corpus: dataclasses.NamedCorpusInfo,
        num_inputs: int,
        num_classes: int,
        epoch: types.EpochType,
        trial_num: types.TrialType = "best",
        prior_scale: float = 0,
        prior_args: Dict = {},
        feature_type: dataclasses.FeatureType = dataclasses.FeatureType.SAMPLES,
        flow_args: Dict = {},
        silence_phone: str = "[SILENCE]",
        register_output: bool = False,
        **kwargs,
    ) -> dataclasses.AlignmentData:
        crp = copy.deepcopy(align_corpus.corpus_info.crp)

        acoustic_mixture_path = mm.CreateDummyMixturesJob(num_classes, num_inputs).out_mixtures

        base_feature_flow = self._make_base_feature_flow(
            align_corpus.corpus_info, feature_type=feature_type, **flow_args
        )

        tf_graph = self._make_tf_graph(
            train_job=train_job.job,
            returnn_config=align_config,
            epoch=epoch,
            trial_num=trial_num,
        )

        checkpoint = self._get_checkpoint(train_job.job, epoch, trial_num=trial_num)
        prior_file = self._get_prior_file(
            train_job=train_job.job,
            prior_config=prior_config,
            checkpoint=checkpoint,
            trial_num=trial_num,
            **prior_args,
        )

        feature_scorer = rasr.PrecomputedHybridFeatureScorer(
            prior_mixtures=acoustic_mixture_path,
            priori_scale=prior_scale,
            prior_file=prior_file,
        )

        feature_flow = self._make_precomputed_tf_feature_flow(
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
        if trial_num is None:
            path = f"nn_align/{align_corpus.name}/{train_job.name}/{exp_full}"
        else:
            path = f"nn_align/{align_corpus.name}/{train_job.name}/trial-{trial_num:03d}/{exp_full}"

        align.set_vis_name(f"Alignment {path}")
        align.add_alias(path)

        if register_output:
            tk.register_output(
                f"{path}.alignment.cache.bundle",
                align.out_alignment_bundle,
            )

        allophone_file = StoreAllophonesJob(crp=crp).out_allophone_file
        state_tying_file = DumpStateTyingJob(crp=crp).out_state_tying

        return dataclasses.AlignmentData(
            alignment_cache_bundle=align.out_alignment_bundle,
            allophone_file=allophone_file,
            state_tying_file=state_tying_file,
            silence_phone=silence_phone,
        )
