__all__ = ["GmmSystem"]

import copy
import itertools
import sys

from typing import Dict, List, Optional, Tuple, Type, Union

# -------------------- Sisyphus --------------------

import sisyphus.global_settings as gs
import sisyphus.toolkit as tk

# -------------------- Recipes --------------------

import i6_core.lexicon.allophones as allophones
import i6_core.am as am
import i6_core.corpus as corpus_recipes
import i6_core.features as features
import i6_core.lda as lda
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
    GmmSatArgs,
    GmmVtlnSatArgs,
    RasrSteps,
)

# -------------------- Init --------------------

Path = tk.setup_path(__package__)

# -------------------- System --------------------


class GmmSystem(RasrSystem):
    """
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

    def __init__(self):
        super().__init__()

        self.monophone_args = None
        self.cart_lda_args = None
        self.triphone_args = None
        self.vtln_args = None
        self.sat_args = None
        self.vtln_sat_args = None

        self.cart_questions = None
        self.cart_trees = {}
        self.lda_matrices = {}
        self.vtln_files = {}

        self.opt_scales = {}

        self.default_align_keep_values = {
            "default": 5,
            "selected": gs.JOB_DEFAULT_KEEP_VALUE,
        }

    # -------------------- Setup --------------------
    def init_system(
        self,
        hybrid_init_args: RasrInitArgs,
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
        self.hybrid_init_args = hybrid_init_args
        self.monophone_args = monophone_args
        self.cart_lda_args = cart_args.cart_lda_args if cart_args is not None else None
        self.triphone_args = triphone_args
        self.vtln_args = vtln_args
        self.sat_args = sat_args
        self.vtln_sat_args = vtln_sat_args

        self._init_am(**self.hybrid_init_args.am_args)

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

        self.cart_questions = (
            cart_args.cart_questions if cart_args is not None else None
        )

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
        feature_flow: Union[
            str, List[str], Tuple[str], rasr.FlagDependentFlowAttribute
        ],
        align_iter: int,
        splits: int,
        accs_per_split: int,
        align_keep_values: Optional[dict] = None,
        **kwargs,
    ):
        if linear_alignment_args is not None:
            self.linear_alignment(
                name,
                corpus_key,
                feature_energy_flow_key,
                prefix=f"{corpus_key}_",
                **linear_alignment_args,
            )

        action_sequence = meta.align_and_accumulate_sequence(
            align_iter, 1, mark_accumulate=False, mark_align=False
        )
        action_sequence += meta.split_and_accumulate_sequence(
            splits, accs_per_split
        ) + ["align!"]

        akv = dict(**self.default_align_keep_values)
        if align_keep_values is not None:
            akv.update(align_keep_values)

        self.train(
            name=name,
            corpus=corpus_key,
            sequence=action_sequence,
            flow=feature_flow,
            initial_mixtures=meta.select_element(
                self.mixtures, corpus_key, "linear_alignment_{}".format(name)
            ),
            align_keep_values=akv,
            **kwargs,
        )
        self.jobs[corpus_key]["train_{}".format(name)].selected_alignment_jobs[
            -1
        ].add_alias("train/{}_{}_align_last".format(corpus_key, name))

        self.jobs[corpus_key]["train_{}".format(name)].selected_mixture_jobs[
            -1
        ].add_alias("train/{}_{}_mix_last".format(corpus_key, name))
        tk.register_output(
            "train/{}_{}_align_bundle_last".format(corpus_key, name),
            self.jobs[corpus_key]["train_{}".format(name)]
            .selected_alignment_jobs[-1]
            .out_alignment_bundle,
        )
        tk.register_output(
            "train/{}_{}_mix_last".format(corpus_key, name),
            self.jobs[corpus_key]["train_{}".format(name)]
            .selected_mixture_jobs[-1]
            .out_mixtures,
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
        for f in self.feature_flows.values():
            f["{}+context".format(context_flow_key)] = lda.add_context_flow(
                feature_net=f[context_flow_key],
                max_size=context_size,
                right=int(context_size / 2.0),
            )

        cart_lda = meta.CartAndLDA(
            original_crp=self.crp[corpus_key],
            initial_flow=self.feature_flows[corpus_key][initial_flow_key],
            context_flow=self.feature_flows[corpus_key][
                "{}+context".format(context_flow_key)
            ],
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
        tk.register_output(
            "{}_{}.tree.xml.gz".format(corpus_key, name), cart_lda.last_cart_tree
        )

        for f in self.feature_flows.values():
            f[
                "{}+context+lda".format(context_flow_key)
            ] = features.add_linear_transform(
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
        feature_flow: Union[
            str, List[str], Tuple[str], rasr.FlagDependentFlowAttribute
        ],
        initial_alignment: Union[
            str, List[str], Tuple[str], rasr.FlagDependentFlowAttribute
        ],
        splits: int,
        accs_per_split: int,
        align_keep_values: Optional[dict] = None,
        **kwargs,
    ):
        action_sequence = (
            ["accumulate"]
            + meta.align_then_split_and_accumulate_sequence(
                splits, accs_per_split, mark_align=False
            )
            + ["align!"]
        )

        akv = dict(**self.default_align_keep_values)
        if align_keep_values is not None:
            akv.update(align_keep_values)

        self.train(
            name=name,
            corpus=corpus_key,
            sequence=action_sequence,
            flow=feature_flow,
            initial_alignment=meta.select_element(
                self.alignments, corpus_key, initial_alignment
            ),
            align_keep_values=akv,
            **kwargs,
        )
        self.jobs[corpus_key]["train_{}".format(name)].selected_alignment_jobs[
            -1
        ].add_alias("train/{}_{}_align_last".format(corpus_key, name))

        self.jobs[corpus_key]["train_{}".format(name)].selected_mixture_jobs[
            -1
        ].add_alias("train/{}_{}_mix_last".format(corpus_key, name))
        tk.register_output(
            "train/{}_{}_align_bundle_last".format(corpus_key, name),
            self.jobs[corpus_key]["train_{}".format(name)]
            .selected_alignment_jobs[-1]
            .out_alignment_bundle,
        )
        tk.register_output(
            "train/{}_{}_mix_last".format(corpus_key, name),
            self.jobs[corpus_key]["train_{}".format(name)]
            .selected_mixture_jobs[-1]
            .out_mixtures,
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
        for c in corpora_keys:
            flow = self.feature_flows[c][base_flow_key]
            if context_size is not None:
                flow = lda.add_context_flow(
                    feature_net=flow,
                    max_size=context_size,
                    right=int(context_size / 2.0),
                )
            if lda_matrix_key is not None:
                flow = features.add_linear_transform(
                    flow, self.lda_matrices[train_corpus_key][lda_matrix_key]
                )
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
        feature_flow_key = self.feature_flows[corpus_key][feature_flow_key]
        warp = vtln.ScoreFeaturesWithWarpingFactorsJob(
            crp=self.crp[corpus_key],
            feature_flow=feature_flow_key,
            feature_scorer=meta.select_element(
                self.feature_scorers, corpus_key, feature_scorer
            ),
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
        for c in eval_corpora_keys:
            self.vtln_features(
                name=name,
                corpus=train_corpus_key,
                raw_feature_flow=self.feature_flows[c][raw_feature_flow_key],
                warping_map=self.vtln_files[train_corpus_key][
                    vtln_files_key + "_warping_map"
                ],
                **kwargs,
            )
            self.feature_flows[c][
                raw_feature_flow_key + "+vtln"
            ] = vtln.recognized_warping_factor_flow(
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
        feature_flow: Union[
            str, List[str], Tuple[str], rasr.FlagDependentFlowAttribute
        ],
        splits: int,
        accs_per_split: int,
        align_keep_values: Optional[dict] = None,
        **kwargs,
    ):
        action_sequence = (
            ["accumulate"]
            + meta.align_then_split_and_accumulate_sequence(splits, accs_per_split)
            + ["align!"]
        )

        akv = dict(**self.default_align_keep_values)
        if align_keep_values is not None:
            akv.update(align_keep_values)

        self.train(
            name=name,
            corpus=corpus_key,
            sequence=action_sequence,
            flow=feature_flow,
            initial_alignment=self.alignments[corpus_key][initial_alignment_key][-1],
            align_keep_values=akv,
            **kwargs,
        )
        self.jobs[corpus_key]["train_{}".format(name)].selected_alignment_jobs[
            -1
        ].add_alias("train/{}_{}_align_last".format(corpus_key, name))

        self.jobs[corpus_key]["train_{}".format(name)].selected_mixture_jobs[
            -1
        ].add_alias("train/{}_{}_mix_last".format(corpus_key, name))
        tk.register_output(
            "train/{}_{}_align_bundle_last".format(corpus_key, name),
            self.jobs[corpus_key]["train_{}".format(name)]
            .selected_alignment_jobs[-1]
            .out_alignment_bundle,
        )
        tk.register_output(
            "train/{}_{}_mix_last".format(corpus_key, name),
            self.jobs[corpus_key]["train_{}".format(name)]
            .selected_mixture_jobs[-1]
            .out_mixtures,
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
        speaker_seg = corpus_recipes.SegmentCorpusBySpeakerJob(
            self.corpora[corpus_key].corpus_file
        )
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
        new_segments = rasr.ClusterMapToSegmentListJob(speaker_seg.out_cluster_map_file)

        overlay_key = (
            "%s_cmllr_%s" % (corpus_key, name) if overlay_key is None else overlay_key
        )
        self.add_overlay(corpus_key, overlay_key)
        self.crp[overlay_key].segment_path = new_segments.out_segment_path
        self.replace_named_flow_attr(
            overlay_key, cache_regex, "cache", mapped_features.out_bundle_path
        )

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
                self.crp[overlay_key].corpus_duration
                / (0.2 * self.crp[overlay_key].concurrent),
                1.0,
            ),
            "cpu": 6,
            "mem": 16,
        }
        self.feature_flows[corpus_key][
            "%s+cmllr" % feature_flow_key
        ] = sat.add_cmllr_transform(
            self.feature_flows[corpus_key][feature_flow_key],
            speaker_seg.out_cluster_map_file,
            cmllr.transforms,
        )

        self.jobs[corpus_key]["segment_corpus_by_speaker"] = speaker_seg
        self.jobs[overlay_key]["mapped_alignment"] = mapped_alignment
        self.jobs[overlay_key]["mapped_features"] = mapped_features
        self.jobs[overlay_key]["new_segments"] = new_segments
        self.jobs[overlay_key]["cmllr"] = cmllr

    @tk.block()
    def sat_training(
        self,
        name: str,
        corpus_key: str,
        feature_cache: Union[
            str, List[str], Tuple[str], rasr.FlagDependentFlowAttribute
        ],
        feature_flow_key: str,
        cache_regex: str,
        alignment: Union[str, List[str], Tuple[str], rasr.FlagDependentFlowAttribute],
        mixtures: Union[str, List[str], Tuple[str], rasr.FlagDependentFlowAttribute],
        splits: int,
        accs_per_split: int,
        align_keep_values: Optional[dict] = None,
        **kwargs,
    ):
        self.estimate_cmllr(
            name=name,
            corpus_key=corpus_key,
            feature_cache=meta.select_element(
                self.feature_caches, corpus_key, feature_cache
            ),
            feature_flow_key=feature_flow_key,
            cache_regex=cache_regex,
            alignment=meta.select_element(self.alignments, corpus_key, alignment),
            mixtures=meta.select_element(self.mixtures, corpus_key, mixtures),
        )

        action_sequence = (
            ["accumulate"]
            + meta.align_then_split_and_accumulate_sequence(
                splits, accs_per_split, mark_align=False
            )
            + ["align!"]
        )

        akv = dict(**self.default_align_keep_values)
        if align_keep_values is not None:
            akv.update(align_keep_values)

        self.train(
            name=name,
            corpus=corpus_key,
            sequence=action_sequence,
            flow="%s+cmllr" % feature_flow_key,
            initial_alignment=meta.select_element(
                self.alignments, corpus_key, alignment
            ),
            align_keep_values=akv,
            **kwargs,
        )
        self.jobs[corpus_key]["train_{}".format(name)].selected_alignment_jobs[
            -1
        ].add_alias("train/{}_{}_align_last".format(corpus_key, name))

        self.jobs[corpus_key]["train_{}".format(name)].selected_mixture_jobs[
            -1
        ].add_alias("train/{}_{}_mix_last".format(corpus_key, name))
        tk.register_output(
            "train/{}_{}_align_bundle_last".format(corpus_key, name),
            self.jobs[corpus_key]["train_{}".format(name)]
            .selected_alignment_jobs[-1]
            .out_alignment_bundle,
        )
        tk.register_output(
            "train/{}_{}_mix_last".format(corpus_key, name),
            self.jobs[corpus_key]["train_{}".format(name)]
            .selected_mixture_jobs[-1]
            .out_mixtures,
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
        feature_flow: Union[
            str, List[str], Tuple[str], rasr.FlagDependentFlowAttribute
        ],
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
        assert (
            "lm_scale" not in kwargs
        ), "please use lm_scales for GmmSystem.recognition()"
        with tk.block(f"{name}_recognition"):
            recog_func = self.recog_and_optimize if optimize_am_lm_scale else self.recog

            pronunciation_scales = (
                [pronunciation_scales]
                if isinstance(pronunciation_scales, float)
                else pronunciation_scales
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
        prev_ctm: str,
        feature_cache: Union[
            str, List[str], Tuple[str], rasr.FlagDependentFlowAttribute
        ],
        cache_regex: str,
        cmllr_mixtures: Union[
            str, List[str], Tuple[str], rasr.FlagDependentFlowAttribute
        ],
        train_corpus_key: str,
        name: str,
        iters: List[int],
        lm_scales: Union[float, List[float]],
        feature_scorer_key: Tuple[str, str],
        optimize_am_lm_scale: bool,
        corpus: str,
        feature_flow: str,
        pronunciation_scales: Union[float, List[float]],
        search_parameters: dict,
        rtf: float,
        mem: float,
        parallelize_conversion: bool,
        lattice_to_ctm_kwargs: dict,
        **kwargs,
    ):
        recog_func = self.recog_and_optimize if optimize_am_lm_scale else self.recog

        pronunciation_scales = (
            [pronunciation_scales]
            if isinstance(pronunciation_scales, float)
            else pronunciation_scales
        )

        lm_scales = [lm_scales] if isinstance(lm_scales, float) else lm_scales

        for it, p, l in itertools.product(iters, pronunciation_scales, lm_scales):
            recognized_corpus = corpus_recipes.ReplaceTranscriptionFromCtmJob(
                self.corpora[corpus].corpus_file,
                self.ctm_files[corpus][
                    f"recog_{train_corpus_key}-{prev_ctm}-{corpus}-ps{p:02.2f}-lm{l:02.2f}-iter{it:02d}"
                ],
            )
            speaker_seq = corpus_recipes.SegmentCorpusBySpeakerJob(
                self.corpora[corpus].corpus_file
            )

            overlay_key = f"{corpus}_it{it}_ps{p}_lm{l}_sat"
            self.add_overlay(corpus, overlay_key)
            self.crp[overlay_key].corpus_config = copy.deepcopy(
                self.crp[corpus].corpus_config
            )
            self.crp[
                overlay_key
            ].corpus_config.file = recognized_corpus.output_corpus_path
            self.crp[overlay_key].segment_path = copy.deepcopy(
                self.crp[corpus].segment_path
            )

            self.corpora[overlay_key] = copy.deepcopy(self.corpora[corpus])
            self.corpora[overlay_key].corpus_file = recognized_corpus.output_corpus_path

            alignment = mm.AlignmentJob(
                crp=self.crp[overlay_key],
                feature_flow=self.feature_flows[overlay_key][feature_flow],
                feature_scorer=self.default_mixture_scorer(
                    meta.select_element(
                        self.mixtures, corpus, (train_corpus_key, cmllr_mixtures)
                    ),
                ),
            )

            self.estimate_cmllr(
                name=name,
                corpus_key=overlay_key,
                feature_cache=meta.select_element(
                    self.feature_caches, corpus, feature_cache
                ),
                feature_flow_key=feature_flow,
                cache_regex=cache_regex,
                alignment=alignment.out_alignment_path,
                mixtures=meta.select_element(
                    self.mixtures, corpus, (train_corpus_key, cmllr_mixtures)
                ),
                overlay_key=overlay_key,
            )
            self.feature_flows[corpus][
                "%s+cmllr" % feature_flow
            ] = sat.add_cmllr_transform(
                feature_net=self.feature_flows[corpus][feature_flow],
                map_file=speaker_seq.out_cluster_map_file,
                transform_dir=self.jobs[overlay_key]["cmllr"].transforms,
            )

            with tk.block(f"{name}_recognition"):
                recog_func(
                    name=f"{name}-{corpus}-ps{p:02.2f}-lm{l:02.2f}-iter{it:02d}",
                    prefix=f"recognition/{name}/",
                    corpus=corpus,
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

    # -------------------- run functions  --------------------

    def run_monophone_step(self, step_args):
        for trn_c in self.train_corpora:
            self.monophone_training(
                corpus_key=trn_c,
                linear_alignment_args=step_args.linear_alignment_args,
                **step_args.training_args,
            )

            for dev_c in self.dev_corpora:
                name = step_args.training_args["name"]
                feature_scorer = (trn_c, f"train_{name}")

                self.recognition(
                    name=f"{trn_c}-{name}",
                    corpus_key=dev_c,
                    feature_scorer_key=feature_scorer,
                    **step_args.recognition_args,
                )

            for tst_c in self.test_corpora:
                name = step_args.training_args["name"]
                feature_scorer = (trn_c, f"train_{name}")

                self.recognition(
                    name=f"{trn_c}-{name}",
                    corpus_key=tst_c,
                    feature_scorer_key=feature_scorer,
                    **step_args.recognition_args,
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

            for dev_c in self.dev_corpora:
                name = step_args.training_args["name"]
                feature_scorer = (trn_c, f"train_{name}")

                self.recognition(
                    f"{trn_c}-{name}",
                    corpus_key=dev_c,
                    feature_scorer_key=feature_scorer,
                    **step_args.recognition_args,
                )

            for tst_c in self.test_corpora:
                name = step_args.training_args["name"]
                feature_scorer = (trn_c, f"train_{name}")

                self.recognition(
                    name=f"{trn_c}-{name}",
                    corpus_key=tst_c,
                    feature_scorer_key=feature_scorer,
                    **step_args.recognition_args,
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

            for dev_c in self.dev_corpora:
                name = step_args.training_args["train"]["name"]
                feature_scorer = (trn_c, f"train_{name}")

                self.recognition(
                    name=f"{trn_c}-{name}",
                    corpus_key=dev_c,
                    feature_scorer_key=feature_scorer,
                    **step_args.recognition_args,
                )

            for tst_c in self.test_corpora:
                pass

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

            for dev_c in self.dev_corpora:
                name = step_args.training_args["name"]
                feature_scorer = (trn_c, f"train_{name}")

                self.sat_recognition(
                    name=f"{trn_c}-{name}",
                    corpus=dev_c,
                    train_corpus_key=trn_c,
                    feature_scorer_key=feature_scorer,
                    **step_args.recognition_args,
                )

            for tst_c in self.test_corpora:
                pass

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

            for dev_c in self.dev_corpora:
                name = step_args.training_args["name"]
                feature_scorer = (trn_c, f"train_{name}")

                self.sat_recognition(
                    name=f"{trn_c}-{name}",
                    corpus=dev_c,
                    train_corpus_key=trn_c,
                    feature_scorer_key=feature_scorer,
                    **step_args.recognition_args,
                )

            for tst_c in self.test_corpora:
                pass

            # ---------- SDM VTLN+SAT ----------
            if step_args.sdm_args is not None:
                self.single_density_mixtures(
                    corpus_key=trn_c,
                    **step_args.sdm_args,
                )

    # -------------------- run setup  --------------------

    def run(self, steps: Union[List[str], RasrSteps]):
        """
        order is important!
        if list: the parameters passed to function "init_system" will be used
        allowed steps: extract, mono, cart, tri, vtln, sat, vtln+sat, forced_align
        step name string must have an allowed step as prefix
        """
        if isinstance(steps, List):
            steps_tmp = steps
            steps = RasrSteps()
            for s in steps_tmp:
                if s == "extract":
                    steps.add_step(s, self.hybrid_init_args.feature_extraction_args)
                elif s == "mono":
                    steps.add_step(s, self.monophone_args)
                elif s == "cart":
                    steps.add_step(
                        s,
                        GmmCartArgs(
                            cart_questions=self.cart_questions,
                            cart_lda_args=self.cart_lda_args,
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
            print(
                "init needs to be run manually. provide: gmm_args, {train,dev,test}_inputs"
            )
            sys.exit(-1)

        # ---------- Corpus statistics, allophones, scoring: stm and sclite ----------
        for all_c in self.train_corpora + self.dev_corpora + self.test_corpora:
            costa_args = copy.deepcopy(self.hybrid_init_args.costa_args)
            if self.crp[all_c].language_model_config is None:
                costa_args["eval_lm"] = False
            self.costa(all_c, prefix="costa/", **costa_args)
            if costa_args["eval_lm"]:
                self.jobs[all_c]["costa"].update_rqmt("run", {"mem": 8, "time": 24})

        for trn_c in self.train_corpora:
            self.store_allophones(trn_c)

        for eval_c in self.dev_corpora + self.test_corpora:
            self.create_stm_from_corpus(eval_c)
            if self.hybrid_init_args.scorer == "kaldi":
                self.set_kaldi_scorer(
                    corpus=eval_c,
                    mapping={"[SILENCE]": ""},
                )
            elif self.hybrid_init_args.scorer == "hub5":
                self.set_hub5_scorer(corpus=eval_c)
            else:
                self.set_sclite_scorer(
                    corpus=eval_c,
                    sort_files=False,
                )

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
                for trn_c in self.train_corpora:
                    self.forced_align(
                        feature_scorer_corpus_key=trn_c,
                        **step_args,
                    )
