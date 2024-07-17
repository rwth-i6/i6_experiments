import copy
import itertools
from typing import Dict, List, Optional

from i6_core import mm, rasr, recognition
from sisyphus import tk

from i6_experiments.users.berger.recipe import returnn

from ... import dataclasses, types
from ..base import RecognitionFunctor
from ..optuna_rasr_base import OptunaRasrFunctor


class OptunaAdvancedTreeSearchFunctor(
    RecognitionFunctor[returnn.OptunaReturnnTrainingJob, returnn.OptunaReturnnConfig],
    OptunaRasrFunctor,
):
    def __call__(
        self,
        train_job: dataclasses.NamedTrainJob[returnn.OptunaReturnnTrainingJob],
        prior_config: returnn.OptunaReturnnConfig,
        recog_config: dataclasses.NamedConfig[returnn.OptunaReturnnConfig],
        recog_corpus: dataclasses.NamedRasrDataInput,
        num_classes: int,
        epochs: List[types.EpochType],
        lm_scales: List[float],
        trial_nums: List[int],
        prior_scales: Optional[List[float]] = None,
        pronunciation_scales: Optional[List[float]] = None,
        prior_args: Optional[Dict] = None,
        am_args: Optional[Dict] = None,
        lattice_to_ctm_kwargs: Optional[Dict] = None,
        feature_type: dataclasses.FeatureType = dataclasses.FeatureType.SAMPLES,
        flow_args: Optional[Dict] = None,
        **kwargs,
    ) -> List[Dict]:
        if prior_scales is None:
            prior_scales = [0.0]
        if pronunciation_scales is None:
            pronunciation_scales = [0.0]
        if prior_args is None:
            prior_args = {}
        if am_args is None:
            am_args = {}
        if lattice_to_ctm_kwargs is None:
            lattice_to_ctm_kwargs = {}
        if flow_args is None:
            flow_args = {}

        assert recog_corpus is not None
        crp = recog_corpus.data.get_crp(
            rasr_python_exe=self.rasr_python_exe,
            rasr_binary_path=self.rasr_binary_path,
            returnn_python_exe=self.returnn_python_exe,
            returnn_root=self.returnn_root,
            blas_lib=self.blas_lib,
            am_args=am_args,
        )
        assert recog_corpus.data.scorer is not None

        acoustic_mixture_path = mm.CreateDummyMixturesJob(num_classes, 1).out_mixtures

        base_feature_flow = self._make_base_feature_flow(recog_corpus.data, feature_type=feature_type, **flow_args)

        recog_results = []

        for (
            lm_scale,
            prior_scale,
            pronunciation_scale,
            epoch,
            trial_num,
        ) in itertools.product(lm_scales, prior_scales, pronunciation_scales, epochs, trial_nums):
            tf_graph = self._make_tf_graph(
                train_job=train_job.job,
                returnn_config=recog_config.config,
                epoch=epoch,
                trial_num=trial_num,
            )
            checkpoint = self._get_checkpoint(train_job.job, epoch, trial_num)
            prior_file = self._get_prior_file(
                train_job=train_job.job,
                prior_config=prior_config,
                checkpoint=checkpoint,
                trial_num=trial_num,
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

            exp_full = f"{recog_config.name}_e-{self._get_epoch_string(epoch)}_pron-{pronunciation_scale:02.2f}_prior-{prior_scale:02.2f}_lm-{lm_scale:02.2f}"
            if trial_num is None:
                path = f"nn_recog/{recog_corpus.name}/{train_job.name}/{exp_full}"
            else:
                path = f"nn_recog/{recog_corpus.name}/{train_job.name}/trial-{trial_num:03d}/{exp_full}"
            rec.set_vis_name(f"Recog {path}")
            rec.add_alias(path)

            scorer_job = self._lattice_scoring(
                crp=crp,
                lattice_bundle=rec.out_lattice_bundle,
                scorer=recog_corpus.data.scorer,
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
                    dataclasses.SummaryKey.TRIAL.value: trial_num,
                    dataclasses.SummaryKey.EPOCH.value: self._get_epoch_value(train_job.job, epoch, trial_num),
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
