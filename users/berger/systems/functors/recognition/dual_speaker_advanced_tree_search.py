import copy
import itertools
from typing import Dict, List

from i6_core import mm, rasr, recognition, returnn
from i6_private.users.vieting.jobs.scoring import (
    CtmRepeatForSpeakersJob,
    MinimumPermutationCtmJob,
)
from sisyphus import tk
from ..base import RecognitionFunctor
from ..rasr_base import RasrFunctor
from ... import dataclasses
from ... import types


class DualSpeakerAdvancedTreeSearchFunctor(
    RecognitionFunctor[returnn.ReturnnTrainingJob, dataclasses.DualSpeakerReturnnConfig],
    RasrFunctor,
):
    def _multi_lattice_scoring(
        self,
        crp: rasr.CommonRasrParameters,
        lattice_bundles: Dict[int, tk.Path],
        scorer: dataclasses.ScorerInfo,
        **kwargs,
    ) -> types.ScoreJob:
        lat2ctms = {
            s_idx: recognition.LatticeToCtmJob(crp=crp, lattice_cache=lattice_bundle, **kwargs).out_ctm_file
            for s_idx, lattice_bundle in lattice_bundles.items()
        }

        ctm_files = {
            s_idx: CtmRepeatForSpeakersJob(lat2ctm, len(lattice_bundles)).out_ctm_file
            for s_idx, lat2ctm in lat2ctms.items()
        }

        scoring_reports = {
            s_idx: scorer.get_score_job(ctm_file).out_report_dir for s_idx, ctm_file in ctm_files.items()
        }

        min_perm_ctm = MinimumPermutationCtmJob(
            scoring_files=scoring_reports, ctms=lat2ctms, stm=scorer.ref_file
        ).out_ctm_file

        score_job = scorer.get_score_job(min_perm_ctm)

        return score_job

    def __call__(
        self,
        train_job: dataclasses.NamedTrainJob[returnn.ReturnnTrainingJob],
        prior_config: dataclasses.DualSpeakerReturnnConfig,
        recog_config: dataclasses.NamedConfig[dataclasses.DualSpeakerReturnnConfig],
        recog_corpus: dataclasses.NamedCorpusInfo,
        num_classes: int,
        epochs: List[types.EpochType],
        lm_scales: List[float],
        prior_scales: List[float] = [0],
        pronunciation_scales: List[float] = [0],
        prior_args: Dict = {},
        lattice_to_ctm_kwargs: Dict = {},
        feature_type: dataclasses.FeatureType = dataclasses.FeatureType.SAMPLES,
        flow_args: Dict = {},
        **kwargs,
    ) -> List[Dict]:
        assert recog_corpus is not None
        crp = copy.deepcopy(recog_corpus.corpus_info.crp)
        assert recog_corpus.corpus_info.scorer is not None

        acoustic_mixture_path = mm.CreateDummyMixturesJob(num_classes, 1).out_mixtures

        base_feature_flow = self._make_base_feature_flow(
            recog_corpus.corpus_info, feature_type=feature_type, **flow_args
        )

        recog_results = []

        for lm_scale, prior_scale, pronunciation_scale, epoch in itertools.product(
            lm_scales, prior_scales, pronunciation_scales, epochs
        ):
            exp_full = f"{recog_config.name}_e-{self._get_epoch_string(epoch)}_pron-{pronunciation_scale:02.2f}_prior-{prior_scale:02.2f}_lm-{lm_scale:02.2f}"
            path = f"nn_recog/{recog_corpus.name}/{train_job.name}/{exp_full}"

            lattice_bundles = {}
            for speaker_idx in [0, 1]:
                tf_graph = self._make_tf_graph(
                    train_job=train_job.job,
                    returnn_config=recog_config.config.get_config_for_speaker(speaker_idx),
                    epoch=epoch,
                )
                checkpoint = self._get_checkpoint(train_job.job, epoch)
                prior_file = self._get_prior_file(
                    prior_config=prior_config.get_config_for_speaker(speaker_idx),
                    checkpoint=checkpoint,
                    **prior_args,
                )

                crp.language_model_config.scale = lm_scale  # type: ignore

                feature_scorer = rasr.PrecomputedHybridFeatureScorer(
                    prior_mixtures=acoustic_mixture_path,
                    priori_scale=prior_scale,
                    prior_file=prior_file,
                )

                model_combination_config = rasr.RasrConfig()
                model_combination_config.pronunciation_scale = pronunciation_scale

                feature_flow = self._make_precomputed_tf_feature_flow(
                    base_feature_flow,
                    tf_graph,
                    checkpoint,
                )

                rec = recognition.AdvancedTreeSearchJob(
                    crp=crp,
                    feature_flow=feature_flow,
                    feature_scorer=feature_scorer,
                    model_combination_config=model_combination_config,
                    **kwargs,
                )

                speaker_path = f"{path}_speaker-{speaker_idx}"

                rec.set_vis_name(f"Recog {speaker_path}")
                rec.add_alias(speaker_path)

                lattice_bundles[speaker_idx] = rec.out_lattice_bundle

            scorer_job = self._multi_lattice_scoring(
                crp=crp,
                lattice_bundles=lattice_bundles,
                scorer=recog_corpus.corpus_info.scorer,
                **lattice_to_ctm_kwargs,
            )
            tk.register_output(
                f"{path}.reports",
                scorer_job.out_report_dir,
            )

            recog_results.append(
                {
                    dataclasses.SummaryKey.TRAIN_NAME.value: train_job.name,
                    dataclasses.SummaryKey.RECOG_NAME.value: recog_config.name,
                    dataclasses.SummaryKey.CORPUS.value: recog_corpus.name,
                    dataclasses.SummaryKey.EPOCH.value: self._get_epoch_value(train_job.job, epoch),
                    dataclasses.SummaryKey.PRON.value: pronunciation_scale,
                    dataclasses.SummaryKey.PRIOR.value: prior_scale,
                    dataclasses.SummaryKey.LM.value: lm_scale,
                    dataclasses.SummaryKey.WER.value: scorer_job.out_wer,
                    dataclasses.SummaryKey.SUB.value: scorer_job.out_percent_substitution,
                    dataclasses.SummaryKey.DEL.value: scorer_job.out_percent_deletions,
                    dataclasses.SummaryKey.INS.value: scorer_job.out_percent_insertions,
                    dataclasses.SummaryKey.ERR.value: scorer_job.out_num_errors,
                }
            )

        return recog_results
