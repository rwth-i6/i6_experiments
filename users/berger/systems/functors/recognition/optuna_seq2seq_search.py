import copy
import itertools
from typing import Dict, List, Optional, Union

from i6_experiments.users.berger.recipe import rasr as custom_rasr
from i6_experiments.users.berger.recipe import recognition, returnn
from i6_experiments.users.berger.recipe.returnn.training import Backend
from sisyphus import tk
from i6_experiments.users.berger.systems.functors.rasr_base import RecognitionScoringType
from ..base import RecognitionFunctor
from ..optuna_rasr_base import OptunaRasrFunctor
from ..seq2seq_base import Seq2SeqFunctor
from ... import dataclasses
from ... import types


class OptunaSeq2SeqSearchFunctor(
    RecognitionFunctor[returnn.OptunaReturnnTrainingJob, returnn.OptunaReturnnConfig],
    Seq2SeqFunctor,
    OptunaRasrFunctor,
):
    def __call__(
        self,
        train_job: dataclasses.NamedTrainJob[returnn.OptunaReturnnTrainingJob],
        prior_config: returnn.OptunaReturnnConfig,
        recog_config: dataclasses.NamedConfig[
            Union[returnn.OptunaReturnnConfig, dataclasses.EncDecConfig[returnn.OptunaReturnnConfig]]
        ],
        recog_corpus: dataclasses.NamedCorpusInfo,
        lookahead_options: Dict,
        epochs: List[types.EpochType],
        trial_nums: List[int],
        lm_scales: List[float] = [0],
        prior_scales: List[float] = [0],
        prior_args: Dict = {},
        lattice_to_ctm_kwargs: Dict = {},
        label_unit: str = "phoneme",
        label_tree_args: Dict = {},
        label_scorer_type: str = "precomputed-log-posterior",
        label_scorer_args: Dict = {},
        feature_type: dataclasses.FeatureType = dataclasses.FeatureType.SAMPLES,
        flow_args: Dict = {},
        model_flow_args: Dict = {},
        backend: Backend = Backend.TENSORFLOW,
        recognition_scoring_type=RecognitionScoringType.Lattice,
        rqmt_update: Optional[dict] = None,
        search_stats: bool = False,
        seq2seq_v2: bool = False,
        **kwargs,
    ) -> List[Dict]:
        assert recog_corpus is not None
        crp = copy.deepcopy(recog_corpus.corpus_info.crp)
        assert recog_corpus.corpus_info.scorer is not None

        label_tree = custom_rasr.LabelTree(
            label_unit,
            lexicon_config=recog_corpus.corpus_info.data.lexicon,
            **label_tree_args,
        )

        mod_label_scorer_args = copy.deepcopy(label_scorer_args)
        if self.requires_label_file(label_unit):
            mod_label_scorer_args["label_file"] = self._get_label_file(crp)

        base_feature_flow = self._make_base_feature_flow(
            recog_corpus.corpus_info, feature_type=feature_type, **flow_args
        )

        recog_results = []

        out_scores = {trial_num: [] for trial_num in trial_nums}

        for lm_scale, prior_scale, epoch, trial_num in itertools.product(lm_scales, prior_scales, epochs, trial_nums):
            checkpoint = self._get_checkpoint(train_job.job, epoch, trial_num=trial_num, backend=backend)

            crp.language_model_config.scale = lm_scale  # type: ignore

            if label_scorer_args.get("use_prior", False) and prior_scale:
                prior_file = self._get_prior_file(
                    train_job=train_job.job,
                    prior_config=prior_config,
                    checkpoint=checkpoint,
                    trial_num=trial_num,
                    backend=backend,
                    **prior_args,
                )
                mod_label_scorer_args["prior_file"] = prior_file
            else:
                mod_label_scorer_args.pop("prior_file", None)
            mod_label_scorer_args["prior_scale"] = prior_scale

            label_scorer = custom_rasr.LabelScorer(label_scorer_type, **mod_label_scorer_args)

            if backend == Backend.TENSORFLOW:
                assert isinstance(recog_config.config, returnn.OptunaReturnnConfig)
                tf_graph = self._make_tf_graph(
                    train_job=train_job.job,
                    returnn_config=recog_config.config,
                    label_scorer_type=label_scorer_type,
                    epoch=epoch,
                    trial_num=trial_num,
                )
                assert isinstance(checkpoint, returnn.Checkpoint)

                feature_flow = self._get_tf_feature_flow_for_label_scorer(
                    label_scorer=label_scorer,
                    base_feature_flow=base_feature_flow,
                    tf_graph=tf_graph,
                    checkpoint=checkpoint,
                    feature_type=feature_type,
                    **model_flow_args,
                )
            elif backend == Backend.PYTORCH:
                assert isinstance(checkpoint, returnn.PtCheckpoint)
                if isinstance(recog_config.config, returnn.OptunaReturnnConfig):
                    onnx_model = self._make_onnx_model(
                        train_job=train_job.job,
                        returnn_config=recog_config.config,
                        checkpoint=checkpoint,
                        trial_num=trial_num,
                    )
                    feature_flow = self._get_onnx_feature_flow_for_label_scorer(
                        label_scorer=label_scorer,
                        base_feature_flow=base_feature_flow,
                        onnx_model=onnx_model,
                        feature_type=feature_type,
                        **model_flow_args,
                    )
                else:
                    enc_model = self._make_onnx_model(
                        train_job=train_job.job,
                        returnn_config=recog_config.config.encoder_config,
                        checkpoint=checkpoint,
                        trial_num=trial_num,
                    )
                    dec_model = self._make_onnx_model(
                        train_job=train_job.job,
                        returnn_config=recog_config.config.decoder_config,
                        checkpoint=checkpoint,
                        trial_num=trial_num,
                    )
                    feature_flow = self._get_onnx_feature_flow_for_label_scorer(
                        label_scorer=label_scorer,
                        base_feature_flow=base_feature_flow,
                        enc_onnx_model=enc_model,
                        dec_onnx_model=dec_model,
                        feature_type=feature_type,
                        **model_flow_args,
                    )
            else:
                raise NotImplementedError

            if seq2seq_v2:
                rec = recognition.GenericSeq2SeqSearchJobV2(
                    crp=crp,
                    feature_flow=feature_flow,
                    label_scorer=label_scorer,
                    label_tree=label_tree,
                    lookahead_options=lookahead_options,
                    **kwargs,
                )
            else:
                rec = recognition.GenericSeq2SeqSearchJob(
                    crp=crp,
                    feature_flow=feature_flow,
                    label_scorer=label_scorer,
                    label_tree=label_tree,
                    lookahead_options=lookahead_options,
                    **kwargs,
                )

            if rqmt_update is not None:
                rec.rqmt.update(rqmt_update)

            exp_full = f"{recog_config.name}_e-{self._get_epoch_string(epoch)}"
            if prior_scale != 0:
                exp_full += f"_prior-{prior_scale:02.2f}"
            if lm_scale != 0:
                exp_full += f"_lm-{lm_scale:02.2f}"

            path = f"nn_recog/{recog_corpus.name}/{train_job.name}/trial-{trial_num:03d}/{exp_full}"

            rec.set_vis_name(f"Recog {path}")
            rec.add_alias(path)

            scorer_job = self._score_recognition_output(
                recognition_scoring_type=recognition_scoring_type,
                crp=crp,
                lattice_bundle=rec.out_lattice_bundle,
                scorer=recog_corpus.corpus_info.scorer,
                **lattice_to_ctm_kwargs,
            )
            tk.register_output(
                f"{path}.reports",
                scorer_job.out_report_dir,
            )

            out_scores[trial_num].append(
                returnn.OptunaReportIntermediateScoreJob(
                    trial_num=trial_num,
                    step=epoch,
                    score=scorer_job.out_wer,
                    study_name=train_job.job.study_name,
                    study_storage=train_job.job.study_storage,
                ).out_reported_score
            )

            rtf = None
            if search_stats:
                stats_job = recognition.ExtractSeq2SeqSearchStatisticsJob(
                    search_logs=list(rec.out_log_file.values()),
                    corpus_duration_hours=recog_corpus.corpus_info.data.corpus_object.duration,
                )
                rtf = stats_job.overall_rtf

                tk.register_output(f"{path}.rtf", rtf)

            recog_results.append(
                {
                    dataclasses.SummaryKey.TRAIN_NAME.value: train_job.name,
                    dataclasses.SummaryKey.RECOG_NAME.value: recog_config.name,
                    dataclasses.SummaryKey.CORPUS.value: recog_corpus.name,
                    dataclasses.SummaryKey.TRIAL.value: trial_num,
                    dataclasses.SummaryKey.EPOCH.value: self._get_epoch_value(train_job.job, epoch, trial_num),
                    dataclasses.SummaryKey.PRIOR.value: prior_scale,
                    dataclasses.SummaryKey.LM.value: lm_scale,
                    dataclasses.SummaryKey.RTF.value: rtf,
                    dataclasses.SummaryKey.WER.value: scorer_job.out_wer,
                    dataclasses.SummaryKey.SUB.value: scorer_job.out_percent_substitution,
                    dataclasses.SummaryKey.DEL.value: scorer_job.out_percent_deletions,
                    dataclasses.SummaryKey.INS.value: scorer_job.out_percent_insertions,
                    dataclasses.SummaryKey.ERR.value: scorer_job.out_num_errors,
                }
            )

        path = f"nn_recog/{recog_corpus.name}/{train_job.name}/trial-{trial_num:03d}/{exp_full}"

        for trial_num in trial_nums:
            final_score = returnn.OptunaReportFinalScoreJob(
                trial_num=trial_num,
                scores=out_scores[trial_num],
                study_name=train_job.job.study_name,
                study_storage=train_job.job.study_storage,
            ).out_reported_score
            tk.register_output(
                f"optuna/{recog_corpus.name}/{train_job.name}/trial-{trial_num:03d}/best_wer",
                value=final_score,
            )

        return recog_results
