__all__ = ["GmmSystem"]

import copy
import itertools
import sys

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Type, Union

# -------------------- Sisyphus --------------------

import sisyphus.global_settings as gs
import sisyphus.toolkit as tk

# -------------------- Recipes --------------------

import i6_core.lexicon.allophones as allophones
import i6_core.am as am
import i6_core.cart as cart
import i6_core.corpus as corpus_recipes
import i6_core.features as features
import i6_core.lda as lda
import i6_core.lm as lm
import i6_core.meta as meta
import i6_core.mm as mm
import i6_core.sat as sat
import i6_core.rasr as rasr
import i6_core.util as util
import i6_core.vtln as vtln

from .rasr_system import RasrSystem

from .util import (
    RasrDataInput,
    RasrInitArgs,
    GmmMonophoneArgs,
    GmmCartArgs,
    GmmTriphoneArgs,
    GmmVtlnArgs,
    PrevCtm,
    GmmSatArgs,
    GmmVtlnSatArgs,
    RasrSteps,
    GmmOutput,
    RecognitionArgs,
)

# -------------------- Init --------------------

Path = tk.setup_path(__package__)

# -------------------- System --------------------


class GmmSystem(RasrSystem):
    """
    This is very limited, so TODO: docstring

    - 3 corpora types: train, dev and test
    - only train corpora will be aligned
    - dev corpora for tuning
    - test corpora for final eval

    to create beforehand:
    - corpora: name and i6_core.meta.system.Corpus
    - lexicon
    - lm

    settings needed:
    - am
    - lm
    - lexicon
    - feature extraction
    - linear alignement
    - monophone training
    - monophone recognition
    - triphone training
    - triphone recognition
    - vtln training
    - vtln recognition
    - sat training
    - sat recognition
    - vtln+sat training
    - vtln+sat recognition

    corpus_key -> str (basically corpus identifier, corpus_name would also be an option)
    corpus_object -> CorpusObject
    corpus_file -> str (would be filepath)
    corpus -> Path
    """

    def __init__(
        self,
        rasr_binary_path: tk.Path,
        rasr_arch: str = "linux-x86_64-standard",
    ):
        """

        :param rasr_binary_path: path to the rasr binary folder
        :param rasr_arch: RASR compile architecture suffix
        """
        super().__init__(rasr_binary_path=rasr_binary_path, rasr_arch=rasr_arch)

        self.monophone_args: Optional[GmmMonophoneArgs] = None
        self.cart_args: Optional[GmmCartArgs] = None
        self.triphone_args: Optional[GmmTriphoneArgs] = None
        self.vtln_args: Optional[GmmVtlnArgs] = None
        self.sat_args: Optional[GmmSatArgs] = None
        self.vtln_sat_args: Optional[GmmVtlnSatArgs] = None

        self.cart_questions: Optional[Union[Type[cart.BasicCartQuestions], cart.PythonCartQuestions, tk.Path]] = None
        self.cart_trees = {}
        self.lda_matrices = {}
        self.vtln_files = {}

        self.opt_scales = {}

        self.default_align_keep_values = {
            "default": 5,
            "selected": gs.JOB_DEFAULT_KEEP_VALUE,
        }
        self.default_split_keep_values = {
            "default": 5,
            "selected": gs.JOB_DEFAULT_KEEP_VALUE,
        }
        self.default_accumulate_keep_values = {
            "default": 5,
            "selected": gs.JOB_DEFAULT_KEEP_VALUE,
        }

        self.outputs = defaultdict(dict)  # type: Dict[str, Dict[str, GmmOutput]]

    # -------------------- Setup --------------------
    def init_system(
        self,
        rasr_init_args: RasrInitArgs,
        train_data: Dict[str, RasrDataInput],
        dev_data: Dict[str, RasrDataInput],
        test_data: Dict[str, RasrDataInput],
        # set pipeline step args via init or via steps param in run function
        monophone_args: Optional[GmmMonophoneArgs] = None,
        cart_args: Optional[GmmCartArgs] = None,
        triphone_args: Optional[GmmTriphoneArgs] = None,
        vtln_args: Optional[GmmVtlnArgs] = None,
        sat_args: Optional[GmmSatArgs] = None,
        vtln_sat_args: Optional[GmmVtlnSatArgs] = None,
    ):
        """
        TODO: docstring
        :param rasr_init_args:
        :param train_data:
        :param dev_data:
        :param test_data:
        :param monophone_args:
        :param cart_args:
        :param triphone_args:
        :param vtln_args:
        :param sat_args:
        :param vtln_sat_args:
        :return:
        """
        self.rasr_init_args = rasr_init_args
        self.monophone_args = monophone_args
        self.cart_args = cart_args
        self.triphone_args = triphone_args
        self.vtln_args = vtln_args
        self.sat_args = sat_args
        self.vtln_sat_args = vtln_sat_args

        self._init_am(**self.rasr_init_args.am_args)

        self._assert_corpus_name_unique(train_data, dev_data, test_data)

        for corpus_key, rasr_data_input in sorted(train_data.items()):
            add_lm = True if rasr_data_input.lm is not None else False
            self.add_corpus(corpus_key, data=rasr_data_input, add_lm=add_lm)
            self.train_corpora.append(corpus_key)

        for corpus_key, rasr_data_input in sorted(dev_data.items()):
            self.add_corpus(corpus_key, data=rasr_data_input, add_lm=True)
            self.dev_corpora.append(corpus_key)

        for corpus_key, rasr_data_input in sorted(test_data.items()):
            self.add_corpus(corpus_key, data=rasr_data_input, add_lm=True)
            self.test_corpora.append(corpus_key)

        self.cart_questions = cart_args.cart_questions if cart_args is not None else None

        for corpus_key in self.train_corpora:
            self.cart_trees[corpus_key] = {}
            self.lda_matrices[corpus_key] = {}
            self.vtln_files[corpus_key] = {}

    # -------------------- Mono Training --------------------

    @tk.block()
    def monophone_training(
        self,
        name: str,
        corpus_key: str,
        linear_alignment_args: dict,
        feature_energy_flow_key: str,
        feature_flow: Union[str, List[str], Tuple[str], rasr.FlagDependentFlowAttribute],
        align_iter: int,
        splits: int,
        accs_per_split: int,
        align_keep_values: Optional[dict] = None,
        split_keep_values: Optional[dict] = None,
        accum_keep_values: Optional[dict] = None,
        dump_alignment_score_report: bool = False,
        mark_accumulate: bool = False,
        mark_align: bool = False,
        **kwargs,
    ):
        """
        TODO: docstring
        :param name: Name of the step
        :param corpus_key: Corpus key to perform training on
        :param linear_alignment_args: extra arguments for the linear alignment
        :param feature_energy_flow_key:
        :param feature_flow:
        :param align_iter: number of align steps
        :param splits: number of split steps
        :param accs_per_split: number of accumulates per split
        :param align_keep_values: sisyphus keep values for cleaning of alignment jobs
        :param split_keep_values: sisyphus keep values for cleaning of split jobs
        :param accum_keep_values: sisyphus keep values for cleaning of accumulation jobs
        :param dump_alignment_score_report: collect the alignment logs and write the report.
            please do not activate this flag if you already cleaned all alignments, as then all deleted
            jobs will re-run.
        :param mark_accumulate: Passed to split_and_accumulate_sequence, defines accums to be marked
        :param mark_align: Passed to split_and_accumulate_sequence, defines alings to be marked
        :param kwargs: passed to AlignSplitAccumulateSequence
        :return:
        """
        if linear_alignment_args is not None:
            self.linear_alignment(
                name,
                corpus_key,
                feature_energy_flow_key,
                prefix=f"{corpus_key}_",
                **linear_alignment_args,
            )

        action_sequence = meta.align_and_accumulate_sequence(
            align_iter,
            1,
            mark_accumulate=mark_accumulate,
            mark_align=mark_align,
        )
        action_sequence += meta.split_and_accumulate_sequence(splits, accs_per_split) + ["align!"]
        akv = dict(**self.default_align_keep_values)
        if align_keep_values is not None:
            akv.update(align_keep_values)

        skv = dict(**self.default_split_keep_values)
        if split_keep_values is not None:
            skv.update(split_keep_values)

        ackv = dict(**self.default_accumulate_keep_values)
        if accum_keep_values is not None:
            ackv.update(accum_keep_values)

        self.train(
            name=name,
            corpus=corpus_key,
            sequence=action_sequence,
            flow=feature_flow,
            initial_mixtures=meta.select_element(self.mixtures, corpus_key, "linear_alignment_{}".format(name)),
            align_keep_values=akv,
            split_keep_values=skv,
            accumulate_keep_values=ackv,
            alias_path="train/{}_{}_action_sequence".format(corpus_key, name),
            **kwargs,
        )
        self.jobs[corpus_key]["train_{}".format(name)].selected_alignment_jobs[-1].add_alias(
            "train/{}_{}_align_last".format(corpus_key, name)
        )

        self.jobs[corpus_key]["train_{}".format(name)].selected_mixture_jobs[-1].add_alias(
            "train/{}_{}_mix_last".format(corpus_key, name)
        )
        tk.register_output(
            "train/{}_{}_align_bundle_last".format(corpus_key, name),
            self.jobs[corpus_key]["train_{}".format(name)].selected_alignment_jobs[-1].out_alignment_bundle,
        )
        tk.register_output(
            "train/{}_{}_mix_last".format(corpus_key, name),
            self.jobs[corpus_key]["train_{}".format(name)].selected_mixture_jobs[-1].out_mixtures,
        )
        if dump_alignment_score_report:
            tk.register_output(
                "train/{}_{}_alignment_report.txt".format(corpus_key, name),
                self.jobs[corpus_key]["train_{}".format(name)].get_alignment_score_report(),
            )

        state_tying_job = allophones.DumpStateTyingJob(self.crp[corpus_key])
        tk.register_output(
            "{}_{}_state_tying".format(corpus_key, name),
            state_tying_job.out_state_tying,
        )

    # -------------------- CaRT and LDA --------------------

    def cart_and_lda(
        self,
        name: str,
        corpus_key: str,
        initial_flow_key: str,
        context_flow_key: str,
        context_size: int,
        alignment: Union[str, List[str], Tuple[str], rasr.FlagDependentFlowAttribute],
        num_dim: int,
        num_iter: int,
        eigenvalue_args: dict,
        generalized_eigenvalue_args: dict,
        **kwargs,
    ):
        """
        TODO:  docstring

        :param name:
        :param corpus_key:
        :param initial_flow_key:
        :param context_flow_key:
        :param context_size:
        :param alignment:
        :param num_dim:
        :param num_iter:
        :param eigenvalue_args:
        :param generalized_eigenvalue_args:
        :param kwargs:
        :return:
        """
        for f in self.feature_flows.values():
            f["{}+context".format(context_flow_key)] = lda.add_context_flow(
                feature_net=f[context_flow_key],
                max_size=context_size,
                right=int(context_size / 2.0),
            )

        cart_lda = meta.CartAndLDA(
            original_crp=self.crp[corpus_key],
            initial_flow=self.feature_flows[corpus_key][initial_flow_key],
            context_flow=self.feature_flows[corpus_key]["{}+context".format(context_flow_key)],
            alignment=meta.select_element(self.alignments, corpus_key, alignment),
            questions=self.cart_questions,
            num_dim=num_dim,
            num_iter=num_iter,
            eigenvalue_args=eigenvalue_args,
            generalized_eigenvalue_args=generalized_eigenvalue_args,
        )
        self.jobs[corpus_key]["cart_and_lda_{}_{}".format(corpus_key, name)] = cart_lda
        self.lda_matrices[corpus_key][name] = cart_lda.last_lda_matrix
        self.cart_trees[corpus_key][name] = cart_lda.last_cart_tree
        tk.register_output(
            "{}_{}_last_num_cart_labels".format(corpus_key, name),
            cart_lda.last_num_cart_labels,
        )
        tk.register_output("{}_{}.tree.xml.gz".format(corpus_key, name), cart_lda.last_cart_tree)

        for f in self.feature_flows.values():
            f["{}+context+lda".format(context_flow_key)] = features.add_linear_transform(
                f["{}+context".format(context_flow_key)], cart_lda.last_lda_matrix
            )

        for crp in self.crp.values():
            crp.acoustic_model_config.state_tying.type = "cart"
            crp.acoustic_model_config.state_tying.file = cart_lda.last_cart_tree

        state_tying_job = allophones.DumpStateTyingJob(self.crp[corpus_key])
        tk.register_output(
            "{}_{}_state_tying".format(corpus_key, name),
            state_tying_job.out_state_tying,
        )

    # -------------------- Tri Training --------------------

    @tk.block()
    def triphone_training(
        self,
        name: str,
        corpus_key: str,
        feature_flow: Union[str, List[str], Tuple[str], rasr.FlagDependentFlowAttribute],
        initial_alignment: Union[str, List[str], Tuple[str], rasr.FlagDependentFlowAttribute],
        splits: int,
        accs_per_split: int,
        align_keep_values: Optional[dict] = None,
        split_keep_values: Optional[dict] = None,
        accum_keep_values: Optional[dict] = None,
        dump_alignment_score_report=False,
        **kwargs,
    ):
        """
        TODO: docstring

        :param name:
        :param corpus_key:
        :param feature_flow:
        :param initial_alignment:
        :param splits:
        :param accs_per_split:
        :param align_keep_values: sisyphus keep values for cleaning of alignment jobs
        :param split_keep_values: sisyphus keep values for cleaning of split jobs
        :param accum_keep_values: sisyphus keep values for cleaning of accumulation jobs
        :param dump_alignment_score_report: collect the alignment logs and write the report.
            please do not activate this flag if you already cleaned all alignments, as then all deleted
            jobs will re-run.
        :param kwargs: passed to AlignSplitAccumulateSequence
        :return:
        """

        action_sequence = (
            ["accumulate"]
            + meta.align_then_split_and_accumulate_sequence(splits, accs_per_split, mark_align=False)
            + ["align!"]
        )

        akv = dict(**self.default_align_keep_values)
        if align_keep_values is not None:
            akv.update(align_keep_values)

        skv = dict(**self.default_split_keep_values)
        if split_keep_values is not None:
            skv.update(split_keep_values)

        ackv = dict(**self.default_accumulate_keep_values)
        if accum_keep_values is not None:
            ackv.update(accum_keep_values)

        self.train(
            name=name,
            corpus=corpus_key,
            sequence=action_sequence,
            flow=feature_flow,
            initial_alignment=meta.select_element(self.alignments, corpus_key, initial_alignment),
            align_keep_values=akv,
            split_keep_values=skv,
            accumulate_keep_values=ackv,
            alias_path="train/{}_{}_action_sequence".format(corpus_key, name),
            **kwargs,
        )
        self.jobs[corpus_key]["train_{}".format(name)].selected_alignment_jobs[-1].add_alias(
            "train/{}_{}_align_last".format(corpus_key, name)
        )

        self.jobs[corpus_key]["train_{}".format(name)].selected_mixture_jobs[-1].add_alias(
            "train/{}_{}_mix_last".format(corpus_key, name)
        )
        tk.register_output(
            "train/{}_{}_align_bundle_last".format(corpus_key, name),
            self.jobs[corpus_key]["train_{}".format(name)].selected_alignment_jobs[-1].out_alignment_bundle,
        )
        tk.register_output(
            "train/{}_{}_mix_last".format(corpus_key, name),
            self.jobs[corpus_key]["train_{}".format(name)].selected_mixture_jobs[-1].out_mixtures,
        )
        if dump_alignment_score_report:
            tk.register_output(
                "train/{}_{}_alignment_report.txt".format(corpus_key, name),
                self.jobs[corpus_key]["train_{}".format(name)].get_alignment_score_report(),
            )

    # -------------------- Vocal Tract Length Normalization --------------------

    def vtln_feature_flow(
        self,
        name: str,
        train_corpus_key: str,
        corpora_keys: [str],
        base_flow_key: str,
        context_size: Optional[int] = None,
        lda_matrix_key: Optional[str] = None,
    ):
        """
        TODO:  docstring

        :param name:
        :param train_corpus_key:
        :param corpora_keys:
        :param base_flow_key:
        :param context_size:
        :param lda_matrix_key:
        :return:
        """
        for c in corpora_keys:
            flow = self.feature_flows[c][base_flow_key]
            if context_size is not None:
                flow = lda.add_context_flow(
                    feature_net=flow,
                    max_size=context_size,
                    right=int(context_size / 2.0),
                )
            if lda_matrix_key is not None:
                flow = features.add_linear_transform(flow, self.lda_matrices[train_corpus_key][lda_matrix_key])
            self.feature_flows[c][name] = flow

    @tk.block()
    def vtln_warping_mixtures(
        self,
        name: str,
        corpus_key: str,
        feature_flow_key: str,
        feature_scorer: Union[str, List[str], Tuple[str], rasr.FeatureScorer],
        alignment: Union[str, List[str], Tuple[str], rasr.FlagDependentFlowAttribute],
        splits: int,
        accs_per_split: int,
    ):
        """
        TODO:  docstring

        :param name:
        :param corpus_key:
        :param feature_flow_key:
        :param feature_scorer:
        :param alignment:
        :param splits:
        :param accs_per_split:
        :return:
        """
        feature_flow_key = self.feature_flows[corpus_key][feature_flow_key]
        warp = vtln.ScoreFeaturesWithWarpingFactorsJob(
            crp=self.crp[corpus_key],
            feature_flow=feature_flow_key,
            feature_scorer=meta.select_element(self.feature_scorers, corpus_key, feature_scorer),
            alignment=meta.select_element(self.alignments, corpus_key, alignment),
        )
        warp.rqmt = {"time": 24, "cpu": 1, "mem": 2}
        self.jobs[corpus_key]["vtln_warping_map_%s" % name] = warp

        seq = meta.TrainWarpingFactorsSequence(
            self.crp[corpus_key],
            None,
            feature_flow_key,
            warp.warping_map,
            warp.alphas_file,
            ["accumulate"] + meta.split_and_accumulate_sequence(splits, accs_per_split),
        )
        self.mixtures[corpus_key]["vtln_warping_mix_%s" % name] = seq.selected_mixtures
        self.vtln_files[corpus_key][name + "_alphas_file"] = warp.alphas_file
        self.vtln_files[corpus_key][name + "_warping_map"] = warp.warping_map
        self.vtln_files[corpus_key][name + "_mixtures"] = seq.selected_mixtures

    @tk.block()
    def extract_vtln_features(
        self,
        name: str,
        train_corpus_key: str,
        eval_corpora_keys: [str],
        raw_feature_flow_key: str,
        vtln_files_key: str,
        **kwargs,
    ):
        """
        TODO: docstring

        :param name:
        :param train_corpus_key:
        :param eval_corpora_keys:
        :param raw_feature_flow_key:
        :param vtln_files_key:
        :param kwargs:
        :return:
        """
        self.vtln_features(
            name=name,
            corpus=train_corpus_key,
            raw_feature_flow=self.feature_flows[train_corpus_key][raw_feature_flow_key],
            warping_map=self.vtln_files[train_corpus_key][vtln_files_key + "_warping_map"],
            **kwargs,
        )
        for c in eval_corpora_keys:
            self.feature_flows[c][raw_feature_flow_key + "+vtln"] = vtln.recognized_warping_factor_flow(
                self.feature_flows[c][raw_feature_flow_key],
                self.vtln_files[train_corpus_key][vtln_files_key + "_alphas_file"],
                self.vtln_files[train_corpus_key][vtln_files_key + "_mixtures"][-1],
            )

    @tk.block()
    def vtln_training(
        self,
        name: str,
        corpus_key: str,
        initial_alignment_key: str,
        feature_flow: Union[str, List[str], Tuple[str], rasr.FlagDependentFlowAttribute],
        splits: int,
        accs_per_split: int,
        align_keep_values: Optional[dict] = None,
        split_keep_values: Optional[dict] = None,
        accum_keep_values: Optional[dict] = None,
        dump_alignment_score_report=False,
        **kwargs,
    ):
        """
        TODO: docstring

        :param name:
        :param corpus_key:
        :param initial_alignment_key:
        :param feature_flow:
        :param splits:
        :param accs_per_split:
        :param align_keep_values: sisyphus keep values for cleaning of alignment jobs
        :param split_keep_values: sisyphus keep values for cleaning of split jobs
        :param accum_keep_values: sisyphus keep values for cleaning of accumulation jobs
        :param dump_alignment_score_report: collect the alignment logs and write the report.
            please do not activate this flag if you already cleaned all alignments, as then all deleted
            jobs will re-run.
        :param kwargs: passed to AlignSplitAccumulateSequence
        :return:
        """
        action_sequence = (
            ["accumulate"] + meta.align_then_split_and_accumulate_sequence(splits, accs_per_split) + ["align!"]
        )

        akv = dict(**self.default_align_keep_values)
        if align_keep_values is not None:
            akv.update(align_keep_values)

        skv = dict(**self.default_split_keep_values)
        if split_keep_values is not None:
            skv.update(split_keep_values)

        ackv = dict(**self.default_accumulate_keep_values)
        if accum_keep_values is not None:
            ackv.update(accum_keep_values)

        self.train(
            name=name,
            corpus=corpus_key,
            sequence=action_sequence,
            flow=feature_flow,
            initial_alignment=self.alignments[corpus_key][initial_alignment_key][-1],
            align_keep_values=akv,
            split_keep_values=skv,
            accumulate_keep_values=ackv,
            alias_path="train/{}_{}_action_sequence".format(corpus_key, name),
            **kwargs,
        )
        self.jobs[corpus_key]["train_{}".format(name)].selected_alignment_jobs[-1].add_alias(
            "train/{}_{}_align_last".format(corpus_key, name)
        )

        self.jobs[corpus_key]["train_{}".format(name)].selected_mixture_jobs[-1].add_alias(
            "train/{}_{}_mix_last".format(corpus_key, name)
        )
        tk.register_output(
            "train/{}_{}_align_bundle_last".format(corpus_key, name),
            self.jobs[corpus_key]["train_{}".format(name)].selected_alignment_jobs[-1].out_alignment_bundle,
        )
        tk.register_output(
            "train/{}_{}_mix_last".format(corpus_key, name),
            self.jobs[corpus_key]["train_{}".format(name)].selected_mixture_jobs[-1].out_mixtures,
        )
        if dump_alignment_score_report:
            tk.register_output(
                "train/{}_{}_alignment_report.txt".format(corpus_key, name),
                self.jobs[corpus_key]["train_{}".format(name)].get_alignment_score_report(),
            )

    # -------------------- Speaker Adaptive Training  --------------------

    def estimate_cmllr(
        self,
        name: str,
        corpus_key: str,
        feature_cache: rasr.FlagDependentFlowAttribute,
        feature_flow_key: str,
        cache_regex: str,
        alignment: rasr.FlagDependentFlowAttribute,
        mixtures: rasr.FlagDependentFlowAttribute,
        overlay_key: Optional[str] = None,
    ):
        """
        TODO: docstring

        :param name:
        :param corpus_key:
        :param feature_cache:
        :param feature_flow_key:
        :param cache_regex:
        :param alignment:
        :param mixtures:
        :param overlay_key:
        :return:
        """
        speaker_seg = corpus_recipes.SegmentCorpusBySpeakerJob(self.corpora[corpus_key].corpus_file)
        old_segment_path = self.crp[corpus_key].segment_path.hidden_paths

        if isinstance(alignment, util.MultiOutputPath):
            alignment = alignment.hidden_paths
        elif isinstance(alignment, rasr.FlagDependentFlowAttribute):
            alignment = alignment.alternatives["task_dependent"].hidden_paths
        else:
            raise Exception("Do not like the type")

        mapped_alignment = rasr.MapSegmentsWithBundlesJob(
            old_segments=old_segment_path,
            cluster_map=speaker_seg.out_cluster_map_file,
            files=alignment,
            filename="cluster.$(TASK).bundle",
        )
        mapped_features = rasr.MapSegmentsWithBundlesJob(
            old_segments=old_segment_path,
            cluster_map=speaker_seg.out_cluster_map_file,
            files=feature_cache.hidden_paths,
            filename="cluster.$(TASK).bundle",
        )

        overlay_key = "%s_cmllr_%s" % (corpus_key, name) if overlay_key is None else overlay_key
        self.add_overlay(corpus_key, overlay_key)
        self.crp[overlay_key].segment_path = speaker_seg.out_segment_path
        self.replace_named_flow_attr(overlay_key, cache_regex, "cache", mapped_features.out_bundle_path)

        cmllr = sat.EstimateCMLLRJob(
            crp=self.crp[overlay_key],
            feature_flow=self.feature_flows[overlay_key][feature_flow_key],
            mixtures=mixtures,
            alignment=mapped_alignment.out_bundle_path,
            cluster_map=speaker_seg.out_cluster_map_file,
            num_clusters=speaker_seg.out_num_speakers,
        )
        cmllr.rqmt = {
            "time": max(
                self.crp[overlay_key].corpus_duration / (0.2 * self.crp[overlay_key].concurrent),
                1.0,
            ),
            "cpu": 6,
            "mem": 16,
        }
        self.feature_flows[corpus_key]["%s+cmllr" % feature_flow_key] = sat.add_cmllr_transform(
            self.feature_flows[corpus_key][feature_flow_key],
            speaker_seg.out_cluster_map_file,
            cmllr.transforms,
        )

        self.jobs[corpus_key]["segment_corpus_by_speaker"] = speaker_seg
        self.jobs[overlay_key]["mapped_alignment"] = mapped_alignment
        self.jobs[overlay_key]["mapped_features"] = mapped_features
        self.jobs[overlay_key]["new_segments"] = speaker_seg
        self.jobs[overlay_key]["cmllr"] = cmllr

    @tk.block()
    def sat_training(
        self,
        name: str,
        corpus_key: str,
        feature_cache: Union[str, List[str], Tuple[str], rasr.FlagDependentFlowAttribute],
        feature_flow_key: str,
        cache_regex: str,
        alignment: Union[str, List[str], Tuple[str], rasr.FlagDependentFlowAttribute],
        mixtures: Union[str, List[str], Tuple[str], rasr.FlagDependentFlowAttribute],
        splits: int,
        accs_per_split: int,
        align_keep_values: Optional[dict] = None,
        split_keep_values: Optional[dict] = None,
        accum_keep_values: Optional[dict] = None,
        dump_alignment_score_report=False,
        **kwargs,
    ):
        """
        TODO: docstring

        :param name:
        :param corpus_key:
        :param feature_cache:
        :param feature_flow_key:
        :param cache_regex:
        :param alignment:
        :param mixtures:
        :param splits:
        :param accs_per_split:
        :param align_keep_values: sisyphus keep values for cleaning of alignment jobs
        :param split_keep_values: sisyphus keep values for cleaning of split jobs
        :param accum_keep_values: sisyphus keep values for cleaning of accumulation jobs
        :param dump_alignment_score_report: collect the alignment logs and write the report.
            please do not activate this flag if you already cleaned all alignments, as then all deleted
            jobs will re-run.
        :param kwargs: passed to AlignSplitAccumulateSequence
        :return:
        """
        self.estimate_cmllr(
            name=name,
            corpus_key=corpus_key,
            feature_cache=meta.select_element(self.feature_caches, corpus_key, feature_cache),
            feature_flow_key=feature_flow_key,
            cache_regex=cache_regex,
            alignment=meta.select_element(self.alignments, corpus_key, alignment),
            mixtures=meta.select_element(self.mixtures, corpus_key, mixtures),
        )

        action_sequence = (
            ["accumulate"]
            + meta.align_then_split_and_accumulate_sequence(splits, accs_per_split, mark_align=False)
            + ["align!"]
        )

        akv = dict(**self.default_align_keep_values)
        if align_keep_values is not None:
            akv.update(align_keep_values)

        skv = dict(**self.default_split_keep_values)
        if split_keep_values is not None:
            skv.update(split_keep_values)

        ackv = dict(**self.default_accumulate_keep_values)
        if accum_keep_values is not None:
            ackv.update(accum_keep_values)

        self.train(
            name=name,
            corpus=corpus_key,
            sequence=action_sequence,
            flow="%s+cmllr" % feature_flow_key,
            initial_alignment=meta.select_element(self.alignments, corpus_key, alignment),
            align_keep_values=akv,
            split_keep_values=skv,
            accumulate_keep_values=ackv,
            alias_path="train/{}_{}_action_sequence".format(corpus_key, name),
            **kwargs,
        )
        self.jobs[corpus_key]["train_{}".format(name)].selected_alignment_jobs[-1].add_alias(
            "train/{}_{}_align_last".format(corpus_key, name)
        )

        self.jobs[corpus_key]["train_{}".format(name)].selected_mixture_jobs[-1].add_alias(
            "train/{}_{}_mix_last".format(corpus_key, name)
        )
        tk.register_output(
            "train/{}_{}_align_bundle_last".format(corpus_key, name),
            self.jobs[corpus_key]["train_{}".format(name)].selected_alignment_jobs[-1].out_alignment_bundle,
        )
        tk.register_output(
            "train/{}_{}_mix_last".format(corpus_key, name),
            self.jobs[corpus_key]["train_{}".format(name)].selected_mixture_jobs[-1].out_mixtures,
        )
        if dump_alignment_score_report:
            tk.register_output(
                "train/{}_{}_alignment_report.txt".format(corpus_key, name),
                self.jobs[corpus_key]["train_{}".format(name)].get_alignment_score_report(),
            )

    # -------------------- recognition  --------------------

    def recognition(
        self,
        name: str,
        iters: List[int],
        lm_scales: Union[float, List[float]],
        feature_scorer_key: Tuple[str, str],
        optimize_am_lm_scale: bool,
        # parameters just for passing through
        corpus_key: str,
        feature_flow: Union[str, List[str], Tuple[str], rasr.FlagDependentFlowAttribute],
        pronunciation_scales: Union[float, List[float]],
        search_parameters: dict,
        rtf: float,
        mem: float,
        parallelize_conversion: bool,
        lattice_to_ctm_kwargs: dict,
        **kwargs,
    ):
        """
        A small wrapper around the meta.System.recog function that will set a Sisyphus block and
        run over all specified model iterations and lm scales.

        :param name: name for the recognition, note that iteration and lm will be named by the function
        :param iters: which training iterations to use for recognition
        :param lm_scales: all lm scales that should be used for recognition
        :param feature_scorer_key: (training_corpus_name, training_name)
        :param optimize_am_lm_scale: will optimize the lm-scale and re-run recognition with the optimal value
        :param kwargs: see meta.System.recog and meta.System.recog_and_optimize
        :param corpus_key: corpus to run recognition on
        :param feature_flow:
        :param pronunciation_scales:
        :param search_parameters:
        :param rtf:
        :param mem:
        :param parallelize_conversion:
        :param lattice_to_ctm_kwargs:
        :return:
        """
        assert "lm_scale" not in kwargs, "please use lm_scales for GmmSystem.recognition()"
        with tk.block(f"{name}_recognition"):
            recog_func = self.recog_and_optimize if optimize_am_lm_scale else self.recog

            pronunciation_scales = (
                [pronunciation_scales] if isinstance(pronunciation_scales, float) else pronunciation_scales
            )

            lm_scales = [lm_scales] if isinstance(lm_scales, float) else lm_scales

            for it, p, l in itertools.product(iters, pronunciation_scales, lm_scales):
                recog_func(
                    name=f"{name}-{corpus_key}-ps{p:02.2f}-lm{l:02.2f}-iter{it:02d}",
                    prefix=f"recognition/{name}/",
                    corpus=corpus_key,
                    flow=feature_flow,
                    feature_scorer=list(feature_scorer_key) + [it - 1],
                    pronunciation_scale=p,
                    lm_scale=l,
                    search_parameters=search_parameters,
                    rtf=rtf,
                    mem=mem,
                    parallelize_conversion=parallelize_conversion,
                    lattice_to_ctm_kwargs=lattice_to_ctm_kwargs,
                    **kwargs,
                )

    def sat_recognition(
        self,
        prev_ctm: PrevCtm,
        feature_cache: Union[str, List[str], Tuple[str], rasr.FlagDependentFlowAttribute],
        cache_regex: str,
        cmllr_mixtures: Union[str, List[str], Tuple[str], rasr.FlagDependentFlowAttribute],
        train_corpus_key: str,
        name: str,
        iters: List[int],
        lm_scales: Union[float, List[float]],
        feature_scorer_key: Tuple[str, str],
        optimize_am_lm_scale: bool,
        corpus_key: str,
        feature_flow: str,
        pronunciation_scales: Union[float, List[float]],
        search_parameters: dict,
        rtf: float,
        mem: float,
        parallelize_conversion: bool,
        lattice_to_ctm_kwargs: dict,
        **kwargs,
    ):
        """
        TODO: docstring

        :param prev_ctm:
        :param feature_cache:
        :param cache_regex:
        :param cmllr_mixtures:
        :param train_corpus_key:
        :param name:
        :param iters:
        :param lm_scales:
        :param feature_scorer_key:
        :param optimize_am_lm_scale:
        :param corpus_key:
        :param feature_flow:
        :param pronunciation_scales:
        :param search_parameters:
        :param rtf:
        :param mem:
        :param parallelize_conversion:
        :param lattice_to_ctm_kwargs:
        :param kwargs:
        :return:
        """
        optlm_string = "-optlm" if prev_ctm.optimized_lm else ""
        prev_ctm_key = f"recog_{train_corpus_key}-{prev_ctm.prev_step_key}-{corpus_key}-ps{prev_ctm.pronunciation_scale:02.2f}-lm{prev_ctm.lm_scale:02.2f}-iter{prev_ctm.iteration:02d}{optlm_string}"
        assert (
            prev_ctm_key in self.ctm_files[corpus_key]
        ), "the previous recognition stage '%s' did not provide the required recognition: %s" % (prev_ctm, prev_ctm_key)
        recognized_corpus = corpus_recipes.ReplaceTranscriptionFromCtmJob(
            self.corpora[corpus_key].corpus_file,
            self.ctm_files[corpus_key][prev_ctm_key],
        )
        speaker_seq = corpus_recipes.SegmentCorpusBySpeakerJob(self.corpora[corpus_key].corpus_file)

        overlay_key = f"{corpus_key}_{name}_ps{prev_ctm.pronunciation_scale:02.2f}-lm{prev_ctm.lm_scale:02.2f}-iter{prev_ctm.iteration:02d}{optlm_string}_sat"
        self.add_overlay(corpus_key, overlay_key)
        self.crp[overlay_key].corpus_config = copy.deepcopy(self.crp[corpus_key].corpus_config)
        self.crp[overlay_key].corpus_config.file = recognized_corpus.output_corpus_path
        self.crp[overlay_key].segment_path = copy.deepcopy(self.crp[corpus_key].segment_path)

        self.corpora[overlay_key] = copy.deepcopy(self.corpora[corpus_key])
        self.corpora[overlay_key].corpus_file = recognized_corpus.output_corpus_path

        alignment = mm.AlignmentJob(
            crp=self.crp[overlay_key],
            feature_flow=self.feature_flows[overlay_key][feature_flow],
            feature_scorer=self.default_mixture_scorer(
                meta.select_element(self.mixtures, corpus_key, (train_corpus_key, cmllr_mixtures)),
            ),
        )

        self.estimate_cmllr(
            name=name,
            corpus_key=overlay_key,
            feature_cache=meta.select_element(self.feature_caches, corpus_key, feature_cache),
            feature_flow_key=feature_flow,
            cache_regex=cache_regex,
            alignment=alignment.out_alignment_path,
            mixtures=meta.select_element(self.mixtures, corpus_key, (train_corpus_key, cmllr_mixtures)),
            overlay_key=overlay_key,
        )
        self.feature_flows[corpus_key]["%s+cmllr" % feature_flow] = sat.add_cmllr_transform(
            feature_net=self.feature_flows[corpus_key][feature_flow],
            map_file=speaker_seq.out_cluster_map_file,
            transform_dir=self.jobs[overlay_key]["cmllr"].transforms,
        )

        with tk.block(f"{name}_recognition"):
            self.recognition(
                name=name,
                iters=iters,
                lm_scales=lm_scales,
                feature_scorer_key=feature_scorer_key,
                optimize_am_lm_scale=optimize_am_lm_scale,
                corpus_key=corpus_key,
                feature_flow=feature_flow + "+cmllr",
                pronunciation_scales=pronunciation_scales,
                search_parameters=search_parameters,
                rtf=rtf,
                mem=mem,
                parallelize_conversion=parallelize_conversion,
                lattice_to_ctm_kwargs=lattice_to_ctm_kwargs,
                **kwargs,
            )

    # -------------------- output helpers  --------------------

    def get_gmm_output(
        self,
        corpus_key: str,
        corpus_type: str,
        step_idx: int,
        steps: RasrSteps,
        extract_features: List[str],
    ):
        """
        :param corpus_key: corpus name identifier
        :param corpus_type: corpus used for: train, dev or test
        :param step_idx: select a specific step from the defined list of steps
        :param steps: all steps in pipeline
        :param extract_features: list of features to extract for later usage
        :return GmmOutput:
        """
        gmm_output = GmmOutput()
        gmm_output.crp = self.crp[corpus_key]
        gmm_output.feature_flows = self.feature_flows[corpus_key]
        gmm_output.features = self.feature_caches[corpus_key]

        for feat_name in extract_features:
            tk.register_output(
                f"features/{corpus_key}_{feat_name}_features.bundle",
                self.feature_bundles[corpus_key][feat_name],
            )

        if corpus_type == "dev" or corpus_type == "test":
            scorer_key = f"estimate_mixtures_sdm.{steps.get_step_names_as_list()[step_idx - 1]}"
            gmm_output.feature_scorers[scorer_key] = self.feature_scorers[corpus_key].get(scorer_key, [None])[-1]
            scorer_key = f"train_{steps.get_step_names_as_list()[step_idx - 1]}"
            gmm_output.feature_scorers[scorer_key] = self.feature_scorers[corpus_key].get(scorer_key, [None])[-1]

        if (corpus_type == "dev" or corpus_type == "test") and self.alignments[corpus_key].get(
            f"alignment_{steps.get_step_names_as_list()[step_idx - 1]}", False
        ):
            gmm_output.alignments = self.alignments[corpus_key][
                f"alignment_{steps.get_step_names_as_list()[step_idx - 1]}"
            ][-1]

        if corpus_type == "train":
            gmm_output.alignments = self.alignments[corpus_key][f"train_{steps.get_prev_gmm_step(step_idx)}"][-1]
            gmm_output.acoustic_mixtures = self.mixtures[corpus_key][f"train_{steps.get_prev_gmm_step(step_idx)}"][-1]

        return gmm_output

    # -------------------- run functions  --------------------

    def run_monophone_step(self, step_args):
        for trn_c in self.train_corpora:
            self.monophone_training(
                corpus_key=trn_c,
                linear_alignment_args=step_args.linear_alignment_args,
                **step_args.training_args,
            )

            name = step_args.training_args["name"]
            feature_scorer = (trn_c, f"train_{name}")

            self.run_recognition_step(
                step_args=step_args,
                name=f"{trn_c}-{name}",
                feature_scorer=feature_scorer,
            )

            # ---------- SDM Mono ----------
            if step_args.sdm_args is not None:
                self.single_density_mixtures(
                    corpus_key=trn_c,
                    **step_args.sdm_args,
                )

    def run_triphone_step(self, step_args):
        for trn_c in self.train_corpora:
            self.triphone_training(
                corpus_key=trn_c,
                **step_args.training_args,
            )

            name = step_args.training_args["name"]
            feature_scorer = (trn_c, f"train_{name}")

            self.run_recognition_step(
                step_args=step_args,
                name=f"{trn_c}-{name}",
                feature_scorer=feature_scorer,
            )

            # ---------- SDM Tri ----------
            if step_args.sdm_args is not None:
                self.single_density_mixtures(
                    corpus_key=trn_c,
                    **step_args.sdm_args,
                )

    def run_vtln_step(self, step_args, step_idx, steps):
        for trn_c in self.train_corpora:
            self.vtln_feature_flow(
                train_corpus_key=trn_c,
                corpora_keys=[trn_c] + self.dev_corpora + self.test_corpora,
                **step_args.training_args["feature_flow"],
            )

            self.vtln_warping_mixtures(
                corpus_key=trn_c,
                feature_flow_key=step_args.training_args["feature_flow"]["name"],
                **step_args.training_args["warp_mix"],
            )

            self.extract_vtln_features(
                name=steps.get_args_via_idx(step_idx - 1).training_args["feature_flow"],
                train_corpus_key=trn_c,
                eval_corpora_keys=self.dev_corpora + self.test_corpora,
                raw_feature_flow_key=step_args.training_args["feature_flow"]["name"],
                vtln_files_key=step_args.training_args["warp_mix"]["name"],
            )

            self.vtln_training(
                corpus_key=trn_c,
                **step_args.training_args["train"],
            )

            name = step_args.training_args["train"]["name"]
            feature_scorer = (trn_c, f"train_{name}")

            self.run_recognition_step(
                step_args=step_args,
                name=f"{trn_c}-{name}",
                feature_scorer=feature_scorer,
            )

            # ---------- SDM VTLN ----------
            if step_args.sdm_args is not None:
                self.single_density_mixtures(
                    corpus_key=trn_c,
                    **step_args.sdm_args,
                )

    def run_sat_step(self, step_args):
        for trn_c in self.train_corpora:
            self.sat_training(
                corpus_key=trn_c,
                **step_args.training_args,
            )

            name = step_args.training_args["name"]
            feature_scorer = (trn_c, f"train_{name}")

            for dev_c in self.dev_corpora:
                self.sat_recognition(
                    name=f"{trn_c}-{name}",
                    corpus_key=dev_c,
                    train_corpus_key=trn_c,
                    feature_scorer_key=feature_scorer,
                    **step_args.recognition_args,
                )

            for tst_c in self.test_corpora:
                recog_args = copy.deepcopy(step_args.recognition_args)
                if step_args.test_recognition_args is None:
                    break
                recog_args.update(step_args.test_recognition_args)
                recog_args["optimize_am_lm_scale"] = False

                self.sat_recognition(
                    name=f"{trn_c}-{name}",
                    corpus_key=tst_c,
                    train_corpus_key=trn_c,
                    feature_scorer_key=feature_scorer,
                    **recog_args,
                )

            # ---------- SDM Sat ----------
            if step_args.sdm_args is not None:
                self.single_density_mixtures(
                    corpus_key=trn_c,
                    **step_args.sdm_args,
                )

    def run_vtln_sat_step(self, step_args):
        for trn_c in self.train_corpora:
            self.sat_training(
                corpus_key=trn_c,
                **step_args.training_args,
            )

            name = step_args.training_args["name"]
            feature_scorer = (trn_c, f"train_{name}")

            for dev_c in self.dev_corpora:
                self.sat_recognition(
                    name=f"{trn_c}-{name}",
                    corpus_key=dev_c,
                    train_corpus_key=trn_c,
                    feature_scorer_key=feature_scorer,
                    **step_args.recognition_args,
                )

            for tst_c in self.test_corpora:
                recog_args = copy.deepcopy(step_args.recognition_args)
                if step_args.test_recognition_args is None:
                    break
                recog_args.update(step_args.test_recognition_args)
                recog_args["optimize_am_lm_scale"] = False

                self.sat_recognition(
                    name=f"{trn_c}-{name}",
                    corpus_key=tst_c,
                    train_corpus_key=trn_c,
                    feature_scorer_key=feature_scorer,
                    **recog_args,
                )

            # ---------- SDM VTLN+SAT ----------
            if step_args.sdm_args is not None:
                self.single_density_mixtures(
                    corpus_key=trn_c,
                    **step_args.sdm_args,
                )

    def run_recognition_step(
        self,
        step_args,
        name: Optional[str] = None,
        feature_scorer: Optional[Tuple[str, str]] = None,
    ):
        assert (name is None and feature_scorer is None and isinstance(step_args, RecognitionArgs)) ^ (
            isinstance(name, str) and isinstance(feature_scorer, Tuple) and not isinstance(step_args, RecognitionArgs)
        ), (
            "please check that variables are not specified in two places. type (name, feature_scorer, step_args):",
            type(name),
            type(feature_scorer),
            type(step_args),
        )

        if isinstance(step_args, RecognitionArgs):
            name = step_args.name
            feature_scorer = step_args.recognition_args.pop("feature_scorer_key")

        for dev_c in self.dev_corpora:
            self.recognition(
                name=name,
                corpus_key=dev_c,
                feature_scorer_key=feature_scorer,
                **step_args.recognition_args,
            )

        for tst_c in self.test_corpora:
            recog_args = copy.deepcopy(step_args.recognition_args)
            if step_args.test_recognition_args is None:
                break
            recog_args.update(step_args.test_recognition_args)
            recog_args["optimize_am_lm_scale"] = False

            self.recognition(
                name=name,
                corpus_key=tst_c,
                feature_scorer_key=feature_scorer,
                **recog_args,
            )

    def run_output_step(self, step_args, step_idx, steps):
        for corpus_key, corpus_type in step_args.corpus_type_mapping.items():
            if corpus_key not in self.train_corpora + self.dev_corpora + self.test_corpora:
                continue
            self.outputs[corpus_key][step_args.name] = self.get_gmm_output(
                corpus_key,
                corpus_type,
                step_idx,
                steps,
                step_args.extract_features,
            )

    # -------------------- run setup  --------------------

    def run(self, steps: Union[List[str], RasrSteps]):
        """
        order is important!
        if list: the parameters passed to function "init_system" will be used
        allowed steps: extract, mono, cart, tri, vtln, sat, vtln+sat, forced_align
        step name string must have an allowed step as prefix

        if not using the run function -> name and corpus almost always need to be added
        """
        if isinstance(steps, List):
            steps_tmp = steps
            steps = RasrSteps()
            for s in steps_tmp:
                if s == "extract":
                    steps.add_step(s, self.rasr_init_args.feature_extraction_args)
                elif s == "mono":
                    steps.add_step(s, self.monophone_args)
                elif s == "cart":
                    steps.add_step(
                        s,
                        GmmCartArgs(
                            cart_questions=self.cart_questions,
                            cart_lda_args=self.cart_args.cart_lda_args,
                        ),
                    )
                elif s == "tri":
                    steps.add_step(s, self.triphone_args)
                elif s == "vtln":
                    steps.add_step(s, self.vtln_args)
                elif s == "sat":
                    steps.add_step(s, self.sat_args)
                elif s == "vtln+sat":
                    steps.add_step(s, self.vtln_sat_args)
                else:
                    raise NotImplementedError
            del steps_tmp

        if "init" in steps.get_step_names_as_list():
            print("init needs to be run manually. provide: gmm_args, {train,dev,test}_inputs")
            sys.exit(-1)

        # ---------- Corpus statistics, allophones, scoring: stm and sclite ----------
        for all_c in self.train_corpora + self.dev_corpora + self.test_corpora:
            costa_args = copy.deepcopy(self.rasr_init_args.costa_args)
            if self.crp[all_c].language_model_config is None:
                costa_args["eval_lm"] = False
            self.costa(all_c, prefix="costa/", **costa_args)
            if costa_args["eval_lm"]:
                self.jobs[all_c]["costa"].update_rqmt("run", {"mem": 8, "time": 24})

        for trn_c in self.train_corpora:
            # TODO: allophones are no longer written into "base" crp,
            # so look out for potential issues
            self.store_allophones(source_corpus=trn_c, target_corpus=trn_c)
            tk.register_output(f"{trn_c}.allophones", self.allophone_files[trn_c])

        self.prepare_scoring()

        for step_idx, (step_name, step_args) in enumerate(steps.get_step_iter()):
            # ---------- Feature Extraction ----------
            if step_name.startswith("extract"):
                self.extract_features(feat_args=step_args)

            # ---------- Monophone ----------
            if step_name.startswith("mono"):
                self.run_monophone_step(step_args)

            # ---------- CaRT ----------
            if step_name.startswith("cart"):
                self.cart_questions = step_args.cart_questions
                for trn_c in self.train_corpora:
                    self.cart_and_lda(
                        corpus_key=trn_c,
                        **step_args.cart_lda_args,
                    )

            # ---------- Triphone ----------
            if step_name.startswith("tri"):
                self.run_triphone_step(step_args)

            # ---------- VTLN ----------
            if step_name.startswith("vtln") and not step_name.startswith("vtln+sat"):
                self.run_vtln_step(
                    step_args=step_args,
                    step_idx=step_idx,
                    steps=steps,
                )

            # ---------- SAT ----------
            if step_name.startswith("sat"):
                self.run_sat_step(step_args)

            # ---------- VTLN+SAT ----------
            if step_name.startswith("vtln+sat"):
                self.run_vtln_sat_step(step_args)

            # ---------- Forced Alignment ----------
            if step_name.startswith("forced_align"):
                corpus_keys = step_args.pop("corpus_keys", self.train_corpora)
                for corpus in corpus_keys:
                    self.forced_align(
                        feature_scorer_corpus_key=corpus,
                        **step_args,
                    )

            # ---------- Only Recognition ----------
            if step_name.startswith("recog"):
                self.run_recognition_step(step_args)

            # ---------- Step Output ----------
            if step_name.startswith("output"):
                self.run_output_step(step_args, step_idx=step_idx, steps=steps)
