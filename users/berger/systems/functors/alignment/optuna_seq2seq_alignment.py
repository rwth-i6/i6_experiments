import copy
from typing import Dict

from sisyphus import tk
from i6_core.lexicon import DumpStateTyingJob, StoreAllophonesJob

from i6_experiments.users.berger.recipe import rasr as custom_rasr
from i6_experiments.users.berger.recipe import mm, returnn

from ... import dataclasses
from ... import types
from ..base import AlignmentFunctor
from ..optuna_rasr_base import OptunaRasrFunctor
from ..seq2seq_base import Seq2SeqFunctor


class OptunaSeq2SeqAlignmentFunctor(
    AlignmentFunctor[returnn.OptunaReturnnTrainingJob, returnn.OptunaReturnnConfig],
    Seq2SeqFunctor,
    OptunaRasrFunctor,
):
    def __call__(
        self,
        train_job: dataclasses.NamedTrainJob[returnn.OptunaReturnnTrainingJob],
        prior_config: returnn.OptunaReturnnConfig,
        align_config: returnn.OptunaReturnnConfig,
        align_corpus: dataclasses.NamedCorpusInfo,
        epoch: types.EpochType,
        trial_num: int,
        prior_scale: float = 0,
        prior_args: Dict = {},
        label_unit: str = "phoneme",
        label_scorer_type: str = "precomputed-log-posterior",
        label_scorer_args: Dict = {},
        feature_type: dataclasses.FeatureType = dataclasses.FeatureType.SAMPLES,
        flow_args: Dict = {},
        silence_phone: str = "<blank>",
        register_output: bool = False,
        **kwargs,
    ) -> dataclasses.AlignmentData:
        crp = copy.deepcopy(align_corpus.corpus_info.crp)

        mod_label_scorer_args = copy.deepcopy(label_scorer_args)
        if self.requires_label_file(label_unit):
            mod_label_scorer_args["label_file"] = self._get_label_file(crp)

        base_feature_flow = self._make_base_feature_flow(
            align_corpus.corpus_info, feature_type=feature_type, **flow_args
        )

        tf_graph = self._make_tf_graph(
            train_job=train_job.job,
            returnn_config=align_config,
            epoch=epoch,
            label_scorer_type=label_scorer_type,
            trial_num=trial_num,
        )

        checkpoint = self._get_checkpoint(train_job=train_job.job, epoch=epoch, trial_num=trial_num)

        if label_scorer_args.get("use_prior", False) and prior_scale:
            prior_file = self._get_prior_file(
                train_job=train_job.job,
                prior_config=prior_config,
                checkpoint=checkpoint,
                trial_num=trial_num,
                **prior_args,
            )
            mod_label_scorer_args["prior_file"] = prior_file
        else:
            mod_label_scorer_args.pop("prior_file", None)
        mod_label_scorer_args["prior_scale"] = prior_scale

        label_scorer = custom_rasr.LabelScorer(label_scorer_type, **mod_label_scorer_args)

        feature_flow = self._get_tf_feature_flow_for_label_scorer(
            label_scorer=label_scorer,
            base_feature_flow=base_feature_flow,
            tf_graph=tf_graph,
            checkpoint=checkpoint,
        )

        align = mm.Seq2SeqAlignmentJob(
            crp=crp,
            feature_flow=feature_flow,
            label_scorer=label_scorer,
            **kwargs,
        )

        exp_full = f"align_e-{self._get_epoch_string(epoch)}_prior-{prior_scale:02.2f}"
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
