import copy
from typing import Dict, Optional

from i6_core import returnn
from i6_core.lexicon.allophones import DumpStateTyingJob, StoreAllophonesJob
from sisyphus import tk

from i6_experiments.users.berger.recipe import mm
from i6_experiments.users.berger.recipe import rasr as custom_rasr
from i6_experiments.users.berger.recipe.returnn.training import Backend, get_backend

from ... import dataclasses, types
from ..base import AlignmentFunctor
from ..seq2seq_base import Seq2SeqFunctor


class Seq2SeqAlignmentFunctor(
    AlignmentFunctor[returnn.ReturnnTrainingJob, returnn.ReturnnConfig],
    Seq2SeqFunctor,
):
    def __call__(
        self,
        train_job: dataclasses.NamedTrainJob[returnn.ReturnnTrainingJob],
        prior_config: returnn.ReturnnConfig,
        align_config: returnn.ReturnnConfig,
        align_corpus: dataclasses.NamedRasrDataInput,
        epoch: types.EpochType,
        am_args: Optional[Dict] = None,
        alias_prefix: str = "align",
        prior_scale: float = 0,
        prior_args: Optional[Dict] = None,
        label_unit: str = "phoneme",
        label_scorer_type: str = "precomputed-log-posterior",
        label_scorer_args: Optional[Dict] = None,
        feature_type: dataclasses.FeatureType = dataclasses.FeatureType.SAMPLES,
        flow_args: Optional[Dict] = None,
        model_flow_args: Optional[Dict] = None,
        silence_phone: str = "<blank>",
        register_output: bool = False,
        **kwargs,
    ) -> dataclasses.AlignmentData:
        if am_args is None:
            am_args = {}
        if prior_args is None:
            prior_args = {}
        if label_scorer_args is None:
            label_scorer_args = {}
        if flow_args is None:
            flow_args = {}
        if model_flow_args is None:
            model_flow_args = {}

        crp = align_corpus.data.get_crp(
            rasr_python_exe=self.rasr_python_exe,
            rasr_binary_path=self.rasr_binary_path,
            returnn_python_exe=self.returnn_python_exe,
            returnn_root=self.returnn_root,
            blas_lib=self.blas_lib,
            am_args=am_args,
        )

        mod_label_scorer_args = copy.deepcopy(label_scorer_args)
        if self.requires_label_file(label_unit):
            mod_label_scorer_args["label_file"] = self._get_label_file(crp)

        base_feature_flow = self._make_base_feature_flow(align_corpus.data, feature_type=feature_type, **flow_args)

        backend = get_backend(align_config)

        checkpoint = self._get_checkpoint(train_job=train_job.job, epoch=epoch)

        if label_scorer_args.get("use_prior", False) and prior_scale:
            prior_file = self._get_prior_file(prior_config=prior_config, checkpoint=checkpoint, **prior_args)
            mod_label_scorer_args["prior_file"] = prior_file
        else:
            mod_label_scorer_args.pop("prior_file", None)
        mod_label_scorer_args["prior_scale"] = prior_scale

        label_scorer = custom_rasr.LabelScorer(label_scorer_type, **mod_label_scorer_args)

        if backend == Backend.TENSORFLOW:
            tf_graph = self._make_tf_graph(
                train_job=train_job.job,
                returnn_config=align_config,
                epoch=epoch,
                label_scorer_type=label_scorer_type,
            )
            assert isinstance(checkpoint, returnn.Checkpoint)

            feature_flow = self._get_tf_feature_flow_for_label_scorer(
                label_scorer=label_scorer,
                base_feature_flow=base_feature_flow,
                tf_graph=tf_graph,
                checkpoint=checkpoint,
                **model_flow_args,
            )
        elif backend == Backend.PYTORCH:
            assert isinstance(checkpoint, returnn.PtCheckpoint)
            onnx_model = self._make_onnx_model(
                returnn_config=align_config,
                checkpoint=checkpoint,
            )
            feature_flow = self._get_onnx_feature_flow_for_label_scorer(
                label_scorer=label_scorer,
                base_feature_flow=base_feature_flow,
                onnx_model=onnx_model,
                feature_type=feature_type,
                **model_flow_args,
            )
        else:
            raise NotImplementedError

        align = mm.Seq2SeqAlignmentJob(
            crp=crp,
            feature_flow=feature_flow,
            label_scorer=label_scorer,
            **kwargs,
        )
        exp_full = f"{alias_prefix}_e-{self._get_epoch_string(epoch)}_prior-{prior_scale:02.2f}"

        path = f"nn_align/{align_corpus.name}/{train_job.name}/{exp_full}"

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
