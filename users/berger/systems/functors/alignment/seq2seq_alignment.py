import copy
import itertools
from typing import Dict, List, Tuple, Union

from i6_core import returnn
from i6_core.lexicon import AddEowPhonemesToLexiconJob
from i6_experiments.users.berger.recipe import rasr as custom_rasr
from i6_experiments.users.berger.recipe import mm
from i6_core.lexicon.allophones import DumpStateTyingJob, StoreAllophonesJob
from sisyphus import tk

from ... import dataclasses
from ... import types
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
        align_corpus: dataclasses.NamedCorpusInfo,
        epochs: List[types.EpochType] = [],
        prior_scales: List[float] = [0],
        prior_args: Dict = {},
        label_unit: str = "phoneme",
        label_scorer_type: str = "precomputed-log-posterior",
        label_scorer_args: Dict = {},
        feature_type: dataclasses.FeatureType = dataclasses.FeatureType.SAMPLES,
        flow_args: Dict = {},
        **kwargs,
    ) -> Union[Dict[Tuple[float, types.EpochType], dataclasses.AlignmentData], dataclasses.AlignmentData,]:
        result = {}

        crp = copy.deepcopy(align_corpus.corpus_info.crp)

        mod_label_scorer_args = copy.deepcopy(label_scorer_args)
        if self.requires_label_file(label_unit):
            mod_label_scorer_args["label_file"] = self._get_label_file(crp)

        base_feature_flow = self._make_base_feature_flow(
            align_corpus.corpus_info, feature_type=feature_type, **flow_args
        )

        for prior_scale, epoch in itertools.product(prior_scales, epochs):
            tf_graph = self._make_tf_graph(
                train_job=train_job.job,
                returnn_config=align_config,
                epoch=epoch,
                label_scorer_type=label_scorer_type,
            )

            checkpoint = self._get_checkpoint(train_job=train_job.job, epoch=epoch)

            if label_scorer_args.get("use_prior", False) and prior_scale:
                prior_file = self._get_prior_file(prior_config=prior_config, checkpoint=checkpoint, **prior_args)
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

            allophone_file = StoreAllophonesJob(crp=crp).out_allophone_file
            state_tying_file = DumpStateTyingJob(crp=crp).out_state_tying
            result[(prior_scale, epoch)] = dataclasses.AlignmentData(
                alignment_cache_bundle=align.out_alignment_bundle,
                allophone_file=allophone_file,
                state_tying_file=state_tying_file,
            )

            exp_full = f"align_e-{self._get_epoch_string(epoch)}_prior-{prior_scale:02.2f}"

            path = f"nn_align/{align_corpus.name}/{train_job.name}/{exp_full}"

            align.set_vis_name(f"Alignment {path}")
            align.add_alias(path)

            tk.register_output(
                f"{path}.alignment.cache.bundle",
                align.out_alignment_bundle,
            )
        if len(result) == 1:
            result = next(iter(result.values()))
        return result
