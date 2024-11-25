import copy
from abc import ABC
from enum import Enum, auto
from functools import partial
from typing import Optional, Union

from i6_core import features, rasr, recognition, returnn
from sisyphus import tk

from i6_experiments.users.berger import helpers
from i6_experiments.users.berger.args.jobs.rasr_init_args import (
    get_feature_extraction_args_8kHz,
    get_feature_extraction_args_16kHz,
)
from i6_experiments.users.berger.helpers.scorer import ScoreJob, ScorerInfo
from i6_experiments.users.berger.recipe import returnn as custom_returnn
from i6_experiments.users.berger.recipe.converse.scoring import (
    MultiChannelCtmToStmJob,
    MultiChannelMultiSegmentCtmToStmJob,
)
from i6_experiments.users.berger.recipe.recognition.scoring import UpsampleCtmFileJob
from i6_experiments.users.berger.recipe.returnn.training import get_backend
from i6_experiments.users.berger.util import lru_cache_with_signature
from i6_experiments.users.berger.recipe.returnn.onnx import ExportPyTorchModelToOnnxJobV2

from .. import dataclasses, types


class RecognitionScoringType(Enum):
    Lattice = auto()
    LatticeUpsample = auto()
    MultiChannelLattice = auto()
    MultiChannelMultiSegmentLattice = auto()


class RasrFunctor(ABC):
    def __init__(
        self,
        returnn_root: tk.Path,
        returnn_python_exe: tk.Path,
        rasr_binary_path: tk.Path,
        rasr_python_exe: tk.Path,
        blas_lib: Optional[tk.Path] = None,
    ) -> None:
        self.returnn_root = returnn_root
        self.returnn_python_exe = returnn_python_exe
        self.rasr_binary_path = rasr_binary_path
        self.rasr_python_exe = rasr_python_exe
        self.blas_lib = blas_lib

    @staticmethod
    def _is_autoregressive_decoding(label_scorer_type: str) -> bool:
        return label_scorer_type != "precomputed-log-posterior"

    @lru_cache_with_signature
    def _get_epoch_value(
        self, train_job: returnn.ReturnnTrainingJob, epoch: types.EpochType
    ) -> Union[int, tk.Variable]:
        if epoch == "best":
            return custom_returnn.GetBestEpochJob(train_job.out_learning_rates).out_epoch
        return epoch

    @lru_cache_with_signature
    def _get_epoch_string(self, epoch: types.EpochType) -> str:
        if isinstance(epoch, str):
            return epoch
        return f"{epoch:03d}"

    def _make_tf_graph(
        self,
        train_job: returnn.ReturnnTrainingJob,
        returnn_config: returnn.ReturnnConfig,
        epoch: types.EpochType,
        label_scorer_type: str = "precomputed-log-posterior",
    ) -> tk.Path:
        rec_step_by_step = "output" if self._is_autoregressive_decoding(label_scorer_type) else None
        graph_compile_job = returnn.CompileTFGraphJob(
            returnn_config,
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe,
            epoch=self._get_epoch_value(train_job, epoch),
            rec_step_by_step=rec_step_by_step,
            rec_json_info=bool(rec_step_by_step),
        )
        return graph_compile_job.out_graph

    def _make_onnx_model(
        self,
        returnn_config: returnn.ReturnnConfig,
        checkpoint: returnn.PtCheckpoint,
        mini_returnn: bool = False,
    ) -> tk.Path:
        if mini_returnn:
            onnx_export_job = ExportPyTorchModelToOnnxJobV2(
                pytorch_checkpoint=checkpoint,
                returnn_config=returnn_config,
                returnn_python_exe=self.returnn_python_exe,
                returnn_root=self.returnn_root,
                verbosity=5,
            )
        else:
            onnx_export_job = returnn.TorchOnnxExportJob(
                returnn_config=returnn_config,
                checkpoint=checkpoint,
                input_names=[],
                output_names=[],
                returnn_python_exe=self.returnn_python_exe,
                returnn_root=self.returnn_root,
            )
        return onnx_export_job.out_onnx_model

    def _make_base_feature_flow(
        self,
        data_input: dataclasses.RasrDataInput,
        feature_type: dataclasses.FeatureType,
        **kwargs,
    ):
        return {
            feature_type.SAMPLES: self._make_base_sample_feature_flow,
            feature_type.GAMMATONE_8K: self._make_base_gt_feature_flow_8k,
            feature_type.GAMMATONE_CACHED_8K: self._make_base_cached_gt_feature_flow_8k,
            feature_type.GAMMATONE_16K: self._make_base_gt_feature_flow_16k,
            feature_type.GAMMATONE_CACHED_16K: self._make_base_cached_gt_feature_flow_16k,
            feature_type.CONCAT_SEC_GAMMATONE_16K: partial(
                self._make_cached_concatenated_gt_feature_flow_16k, use_sec=True, use_mix=False
            ),
            feature_type.CONCAT_MIX_GAMMATONE_16K: partial(
                self._make_cached_concatenated_gt_feature_flow_16k, use_sec=False, use_mix=True
            ),
            feature_type.CONCAT_SEC_MIX_GAMMATONE_16K: partial(
                self._make_cached_concatenated_gt_feature_flow_16k, use_sec=True, use_mix=True
            ),
            feature_type.LOGMEL_16K: self._make_base_logmel_feature_flow_16k,
        }[feature_type](data_input=data_input, **kwargs)

    def _make_base_sample_feature_flow(
        self, data_input: dataclasses.RasrDataInput, dc_detection: bool = False, **kwargs
    ):
        audio_format = data_input.corpus_object.audio_format
        args = {
            "audio_format": audio_format,
            "dc_detection": dc_detection,
            "input_options": {"block-size": 1},
            "scale_input": 2**-15,
        }
        args.update(kwargs)
        return features.samples_flow(**args)

    def _make_base_gt_feature_flow_8k(self, data_input: dataclasses.RasrDataInput, dc_detection: bool = False, **_):
        gt_options = copy.deepcopy(get_feature_extraction_args_8kHz(dc_detection=dc_detection)["gt"]["gt_options"])
        audio_format = data_input.corpus_object.audio_format
        gt_options["samples_options"]["audio_format"] = audio_format
        gt_options["add_features_output"] = True
        return features.gammatone_flow(**gt_options)

    def _make_base_cached_gt_feature_flow_8k(
        self, data_input: dataclasses.RasrDataInput, dc_detection: bool = False, **_
    ):
        crp = data_input.get_crp(
            rasr_python_exe=self.rasr_python_exe,
            rasr_binary_path=self.rasr_binary_path,
            returnn_python_exe=self.returnn_python_exe,
            returnn_root=self.returnn_root,
        )
        gt_job = features.GammatoneJob(crp=crp, **get_feature_extraction_args_8kHz(dc_detection=dc_detection)["gt"])
        feature_path = rasr.FlagDependentFlowAttribute(
            "cache_mode",
            {
                "task_dependent": gt_job.out_feature_path["gt"],
            },
        )
        return features.basic_cache_flow(
            cache_files=feature_path,
        )

    def _make_base_gt_feature_flow_16k(self, data_input: dataclasses.RasrDataInput, dc_detection: bool = False, **_):
        gt_options = copy.deepcopy(get_feature_extraction_args_16kHz(dc_detection=dc_detection)["gt"]["gt_options"])
        audio_format = data_input.corpus_object.audio_format
        gt_options["samples_options"]["audio_format"] = audio_format
        gt_options["add_features_output"] = True
        return features.gammatone_flow(**gt_options)

    def _make_base_cached_gt_feature_flow_16k(
        self, data_input: dataclasses.RasrDataInput, dc_detection: bool = False, **_
    ):
        crp = data_input.get_crp(
            rasr_python_exe=self.rasr_python_exe,
            rasr_binary_path=self.rasr_binary_path,
            returnn_python_exe=self.returnn_python_exe,
            returnn_root=self.returnn_root,
        )
        gt_job = features.GammatoneJob(crp=crp, **get_feature_extraction_args_16kHz(dc_detection=dc_detection)["gt"])
        feature_path = rasr.FlagDependentFlowAttribute(
            "cache_mode",
            {
                "task_dependent": gt_job.out_feature_path["gt"],
            },
        )
        return features.basic_cache_flow(
            cache_files=feature_path,
        )

    def _make_cached_concatenated_gt_feature_flow_16k(
        self,
        data_input: dataclasses.RasrDataInput,
        dc_detection: bool = False,
        use_mix: bool = True,
        use_sec: bool = True,
        **_,
    ):
        # TODO: why does this assert fail?
        # assert isinstance(data_input.corpus_object, SeparatedCorpusObject)
        crp_prim = data_input.get_crp(
            rasr_python_exe=self.rasr_python_exe,
            rasr_binary_path=self.rasr_binary_path,
            returnn_python_exe=self.returnn_python_exe,
            returnn_root=self.returnn_root,
        )

        crp_sec = copy.deepcopy(crp_prim)
        assert crp_sec.corpus_config is not None
        crp_sec.corpus_config.file = data_input.corpus_object.secondary_corpus_file

        crp_mix = copy.deepcopy(crp_prim)
        assert crp_mix.corpus_config is not None
        crp_mix.corpus_config.file = data_input.corpus_object.mix_corpus_file

        cache_files = []
        crp_list = [crp_prim]
        if use_sec:
            crp_list.append(crp_sec)
        if use_mix:
            crp_list.append(crp_mix)

        for crp in crp_list:
            gt_job = features.GammatoneJob(
                crp=crp, **get_feature_extraction_args_16kHz(dc_detection=dc_detection)["gt"]
            )
            feature_path = rasr.FlagDependentFlowAttribute(
                "cache_mode",
                {
                    "task_dependent": gt_job.out_feature_path["gt"],
                },
            )
            cache_files.append(feature_path)

        return features.basic_cache_flow(cache_files=cache_files)

    def _make_base_logmel_feature_flow_16k(
        self, data_input: dataclasses.RasrDataInput, dc_detection: bool = False, **_
    ):
        filterbank_options = copy.deepcopy(
            get_feature_extraction_args_16kHz(dc_detection=dc_detection)["filterbank"]["filterbank_options"]
        )
        audio_format = data_input.corpus_object.audio_format
        filterbank_options["samples_options"]["audio_format"] = audio_format
        filterbank_options["add_features_output"] = True
        return features.filterbank_flow(**filterbank_options)

    @lru_cache_with_signature
    def _get_checkpoint(
        self,
        train_job: returnn.ReturnnTrainingJob,
        epoch: types.EpochType,
    ) -> types.CheckpointType:
        if epoch == "best":
            return custom_returnn.GetBestCheckpointJob(
                model_dir=train_job.out_model_dir,
                learning_rates=train_job.out_learning_rates,
                backend=custom_returnn.get_backend(train_job.returnn_config),
            ).out_checkpoint
        return train_job.out_checkpoints[epoch]

    def _get_prior_file(
        self,
        prior_config: returnn.ReturnnConfig,
        checkpoint: types.CheckpointType,
        **kwargs,
    ) -> tk.Path:
        backend = get_backend(prior_config)
        if backend == backend.TENSORFLOW:
            prior_job = returnn.ReturnnComputePriorJobV2(
                model_checkpoint=checkpoint,
                returnn_config=prior_config,
                returnn_root=self.returnn_root,
                returnn_python_exe=self.returnn_python_exe,
                **kwargs,
            )

            prior_job.update_rqmt("run", {"file_size": 150})

            return prior_job.out_prior_xml_file
        elif backend == backend.PYTORCH:
            forward_job = custom_returnn.ReturnnForwardComputePriorJob(
                model_checkpoint=checkpoint,
                returnn_config=prior_config,
                returnn_root=self.returnn_root,
                returnn_python_exe=self.returnn_python_exe,
                mem_rqmt=8,
            )
            return forward_job.out_prior_xml_file
        else:
            raise NotImplementedError

    @lru_cache_with_signature
    def _get_native_lstm_op(self) -> tk.Path:
        return helpers.get_native_lstm_op(
            returnn_root=self.returnn_root, returnn_python_exe=self.returnn_python_exe, blas_lib=self.blas_lib
        )

    def _make_precomputed_tf_feature_flow(
        self,
        base_flow: rasr.FlowNetwork,
        tf_graph: tk.Path,
        tf_checkpoint: returnn.Checkpoint,
        output_layer_name: str = "output",
        **_,
    ) -> rasr.FlowNetwork:
        # tf flow (model scoring done in tf flow node) #
        input_name = "tf-fwd_input"

        tf_flow = rasr.FlowNetwork()
        tf_flow.add_input(input_name)
        tf_flow.add_output("features")
        tf_flow.add_param("id")

        tf_fwd = tf_flow.add_node("tensorflow-forward", "tf-fwd", {"id": "$(id)"})
        tf_flow.link(f"network:{input_name}", f"{tf_fwd}:input")
        tf_flow.link(f"{tf_fwd}:log-posteriors", "network:features")

        tf_flow.config = rasr.RasrConfig()  # type: ignore
        tf_flow.config[tf_fwd].input_map.info_0.param_name = "input"  # type: ignore
        tf_flow.config[tf_fwd].input_map.info_0.tensor_name = "extern_data/placeholders/data/data"  # type: ignore
        tf_flow.config[tf_fwd].input_map.info_0.seq_length_tensor_name = "extern_data/placeholders/data/data_dim0_size"  # type: ignore

        tf_flow.config[tf_fwd].output_map.info_0.param_name = "log-posteriors"  # type: ignore
        tf_flow.config[tf_fwd].output_map.info_0.tensor_name = f"{output_layer_name}/output_batch_major"  # type: ignore

        tf_flow.config[tf_fwd].loader.type = "meta"  # type: ignore
        tf_flow.config[tf_fwd].loader.meta_graph_file = tf_graph  # type: ignore
        tf_flow.config[tf_fwd].loader.saved_model_file = tf_checkpoint  # type: ignore

        tf_flow.config[tf_fwd].loader.required_libraries = self._get_native_lstm_op()  # type: ignore

        # interconnect flows #
        ext_flow = rasr.FlowNetwork()
        base_mapping = ext_flow.add_net(base_flow)
        tf_mapping = ext_flow.add_net(tf_flow)
        ext_flow.interconnect_inputs(base_flow, base_mapping)
        ext_flow.interconnect(
            base_flow,
            base_mapping,
            tf_flow,
            tf_mapping,
            {list(base_flow.outputs)[0]: input_name},
        )

        ext_flow.interconnect_outputs(tf_flow, tf_mapping)
        # ensure cache_mode as base feature net
        ext_flow.add_flags(base_flow.flags)
        return ext_flow

    def _make_precomputed_onnx_feature_flow(
        self,
        base_flow: rasr.FlowNetwork,
        onnx_model: tk.Path,
        features_name: str = "data",
        features_size_name: str = "data:size1",
        output_name: str = "log_probs",
        **_,
    ) -> rasr.FlowNetwork:
        # tf flow (model scoring done in tf flow node) #
        input_name = "onnx-fwd_input"

        onnx_flow = rasr.FlowNetwork()
        onnx_flow.add_input(input_name)
        onnx_flow.add_output("features")
        onnx_flow.add_param("id")

        onnx_fwd = onnx_flow.add_node("onnx-forward", "onnx-fwd", {"id": "$(id)"})
        onnx_flow.link(f"network:{input_name}", f"{onnx_fwd}:input")
        onnx_flow.link(f"{onnx_fwd}:log-posteriors", "network:features")

        onnx_flow.config = rasr.RasrConfig()  # type: ignore
        onnx_flow.config[onnx_fwd].io_map.features = features_name
        onnx_flow.config[onnx_fwd].io_map.features_size = features_size_name
        onnx_flow.config[onnx_fwd].io_map.output = output_name

        onnx_flow.config[onnx_fwd].session.file = onnx_model
        onnx_flow.config[onnx_fwd].session.inter_op_num_threads = 2
        onnx_flow.config[onnx_fwd].session.intra_op_num_threads = 2

        # interconnect flows #
        ext_flow = rasr.FlowNetwork()
        base_mapping = ext_flow.add_net(base_flow)
        tf_mapping = ext_flow.add_net(onnx_flow)
        ext_flow.interconnect_inputs(base_flow, base_mapping)
        ext_flow.interconnect(
            base_flow,
            base_mapping,
            onnx_flow,
            tf_mapping,
            {list(base_flow.outputs)[0]: input_name},
        )

        ext_flow.interconnect_outputs(onnx_flow, tf_mapping)
        # ensure cache_mode as base feature net
        ext_flow.add_flags(base_flow.flags)
        return ext_flow

    def _lattice_to_ctm_scoring(
        self,
        crp: rasr.CommonRasrParameters,
        lattice_bundle: tk.Path,
        scorer: ScorerInfo,
        **kwargs,
    ) -> ScoreJob:
        lat2ctm = recognition.LatticeToCtmJob(
            crp=crp,
            lattice_cache=lattice_bundle,
            **kwargs,
        )

        score_job = scorer.get_score_job(lat2ctm.out_ctm_file)

        return score_job

    def _upsampled_lattice_to_ctm_scoring(
        self,
        crp: rasr.CommonRasrParameters,
        lattice_bundle: tk.Path,
        scorer: ScorerInfo,
        **kwargs,
    ) -> ScoreJob:
        lat2ctm = recognition.LatticeToCtmJob(
            crp=crp,
            lattice_cache=lattice_bundle,
            **kwargs,
        )
        lat2ctm = UpsampleCtmFileJob(lat2ctm.out_ctm_file)

        score_job = scorer.get_score_job(lat2ctm.out_ctm_file)

        return score_job

    def _multi_channel_multi_segment_lattice_scoring(
        self,
        crp: rasr.CommonRasrParameters,
        lattice_bundle: tk.Path,
        scorer: ScorerInfo,
        **kwargs,
    ) -> ScoreJob:
        lat2ctm = recognition.LatticeToCtmJob(
            crp=crp,
            lattice_cache=lattice_bundle,
            **kwargs,
        )
        ctm_to_stm_job = MultiChannelMultiSegmentCtmToStmJob(lat2ctm.out_ctm_file)

        score_job = scorer.get_score_job(ctm_to_stm_job.out_stm_file)

        return score_job

    def _multi_channel_lattice_scoring(
        self,
        crp: rasr.CommonRasrParameters,
        lattice_bundle: tk.Path,
        scorer: ScorerInfo,
        **kwargs,
    ) -> ScoreJob:
        lat2ctm = recognition.LatticeToCtmJob(
            crp=crp,
            lattice_cache=lattice_bundle,
            **kwargs,
        )
        ctm_to_stm_job = MultiChannelCtmToStmJob(lat2ctm.out_ctm_file)

        score_job = scorer.get_score_job(ctm_to_stm_job.out_stm_file)

        return score_job

    def _score_recognition_output(self, recognition_scoring_type: RecognitionScoringType, **kwargs) -> ScoreJob:
        if recognition_scoring_type == RecognitionScoringType.Lattice:
            return self._lattice_to_ctm_scoring(**kwargs)
        if recognition_scoring_type == RecognitionScoringType.LatticeUpsample:
            return self._upsampled_lattice_to_ctm_scoring(**kwargs)
        if recognition_scoring_type == RecognitionScoringType.MultiChannelLattice:
            return self._multi_channel_lattice_scoring(**kwargs)
        if recognition_scoring_type == RecognitionScoringType.MultiChannelMultiSegmentLattice:
            return self._multi_channel_multi_segment_lattice_scoring(**kwargs)
        raise NotImplementedError
