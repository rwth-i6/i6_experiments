import copy
import itertools
from typing import Dict, List, Optional, Union

from i6_core import returnn
from sisyphus import tk

from i6_experiments.users.berger.recipe import rasr as custom_rasr
from i6_experiments.users.berger.recipe import recognition
from i6_experiments.users.berger.recipe.returnn.training import Backend, get_backend

from ... import dataclasses, types
from ..base import RecognitionFunctor
from ..rasr_base import RecognitionScoringType
from ..seq2seq_base import Seq2SeqFunctor


class Seq2SeqSearchFunctor(
    RecognitionFunctor[returnn.ReturnnTrainingJob, returnn.ReturnnConfig],
    Seq2SeqFunctor,
):
    def __call__(
        self,
        train_job: dataclasses.NamedTrainJob[returnn.ReturnnTrainingJob],
        prior_config: returnn.ReturnnConfig,
        recog_config: dataclasses.NamedConfig[
            Union[returnn.ReturnnConfig, dataclasses.EncDecConfig[returnn.ReturnnConfig]]
        ],
        recog_corpus: dataclasses.NamedRasrDataInput,
        lookahead_options: Dict,
        epochs: List[types.EpochType],
        am_args: Optional[Dict] = None,
        lm_scales: Optional[List[float]] = None,
        prior_scales: Optional[List[float]] = None,
        prior_args: Optional[Dict] = None,
        prior_epoch: Optional[int] = None,
        lattice_to_ctm_kwargs: Optional[Dict] = None,
        label_unit: str = "phoneme",
        label_tree_args: Optional[Dict] = None,
        label_scorer_type: str = "precomputed-log-posterior",
        label_scorer_args: Optional[Dict] = None,
        feature_type: dataclasses.FeatureType = dataclasses.FeatureType.SAMPLES,
        flow_args: Optional[Dict] = None,
        model_flow_args: Optional[Dict] = None,
        recognition_scoring_type=RecognitionScoringType.Lattice,
        rqmt_update: Optional[dict] = None,
        search_stats: bool = False,
        seq2seq_v2: bool = False,
        mini_returnn: bool = False,
        **kwargs,
    ) -> List[Dict]:
        if am_args is None:
            am_args = {}
        if lm_scales is None:
            lm_scales = [0.0]
        if prior_scales is None:
            prior_scales = [0.0]
        if prior_args is None:
            prior_args = {}
        if lattice_to_ctm_kwargs is None:
            lattice_to_ctm_kwargs = {}
        if label_tree_args is None:
            label_tree_args = {}
        if label_scorer_args is None:
            label_scorer_args = {}
        if flow_args is None:
            flow_args = {}
        if model_flow_args is None:
            model_flow_args = {}

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

        label_tree = custom_rasr.LabelTree(
            label_unit,
            lexicon_config=recog_corpus.data.lexicon,
            **label_tree_args,
        )

        mod_label_scorer_args = copy.deepcopy(label_scorer_args)
        if self.requires_label_file(label_unit):
            mod_label_scorer_args["label_file"] = self._get_label_file(crp)

        base_feature_flow = self._make_base_feature_flow(recog_corpus.data, feature_type=feature_type, **flow_args)

        recog_results = []

        if isinstance(recog_config.config, returnn.ReturnnConfig):
            backend = get_backend(recog_config.config)
        else:
            backend = get_backend(recog_config.config.encoder_config)

        for lm_scale, prior_scale, epoch in itertools.product(lm_scales, prior_scales, epochs):
            checkpoint = self._get_checkpoint(train_job.job, epoch)

            crp.language_model_config.scale = lm_scale  # type: ignore

            if label_scorer_args.get("use_prior", False):
                if prior_epoch is None:
                    prior_checkpoint = checkpoint
                else:
                    prior_checkpoint = self._get_checkpoint(train_job.job, prior_epoch)
                prior_file = self._get_prior_file(
                    prior_config=prior_config,
                    checkpoint=prior_checkpoint,
                    **prior_args,
                )
                mod_label_scorer_args["prior_file"] = prior_file
            else:
                mod_label_scorer_args.pop("prior_file", None)
            mod_label_scorer_args["prior_scale"] = prior_scale

            label_scorer = custom_rasr.LabelScorer(label_scorer_type, **mod_label_scorer_args)

            if backend == Backend.TENSORFLOW:
                assert isinstance(recog_config.config, returnn.ReturnnConfig)
                tf_graph = self._make_tf_graph(
                    train_job=train_job.job,
                    returnn_config=recog_config.config,
                    label_scorer_type=label_scorer_type,
                    epoch=epoch,
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
                if isinstance(recog_config.config, returnn.ReturnnConfig):
                    onnx_model = self._make_onnx_model(
                        returnn_config=recog_config.config,
                        checkpoint=checkpoint,
                        mini_returnn=mini_returnn,
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
                        returnn_config=recog_config.config.encoder_config,
                        checkpoint=checkpoint,
                        mini_returnn=mini_returnn,
                    )
                    dec_model = self._make_onnx_model(
                        returnn_config=recog_config.config.decoder_config,
                        checkpoint=checkpoint,
                        mini_returnn=mini_returnn,
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

            path = f"nn_recog/{recog_corpus.name}/{train_job.name}/{exp_full}"

            rec.set_vis_name(f"Recog {path}")
            rec.add_alias(path)

            scorer_job = self._score_recognition_output(
                recognition_scoring_type=recognition_scoring_type,
                crp=crp,
                lattice_bundle=rec.out_lattice_bundle,
                scorer=recog_corpus.data.scorer,
                **lattice_to_ctm_kwargs,
            )

            tk.register_output(
                f"{path}.reports",
                scorer_job.out_report_dir,
            )

            rtf = None
            if search_stats:
                assert recog_corpus.data.corpus_object.duration is not None
                stats_job = recognition.ExtractSeq2SeqSearchStatisticsJob(
                    search_logs=list(rec.out_log_file.values()),
                    corpus_duration_hours=recog_corpus.data.corpus_object.duration,
                )
                rtf = stats_job.overall_rtf

                tk.register_output(f"{path}.rtf", rtf)

            recog_results.append(
                {
                    dataclasses.SummaryKey.TRAIN_NAME.value: train_job.name,
                    dataclasses.SummaryKey.RECOG_NAME.value: recog_config.name,
                    dataclasses.SummaryKey.CORPUS.value: recog_corpus.name,
                    dataclasses.SummaryKey.EPOCH.value: self._get_epoch_value(train_job.job, epoch),
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

        return recog_results
