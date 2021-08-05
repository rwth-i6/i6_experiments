__all__ = ["GmmSystem"]

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
import i6_core.sat as sat
import i6_core.rasr as rasr
import i6_core.util as util
import i6_core.vtln as vtln

from .rasr_system import RasrSystem

from .util import (
    RasrDataInput,
    RasrInitArgs,
    GmmMonophoneArgs,
    GmmTriphoneArgs,
    GmmVtlnArgs,
    GmmSatArgs,
    GmmVtlnSatArgs,
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
    """

    def __init__(self):
        super().__init__()

        self.gmm_monophone_args = None
        self.gmm_triphone_args = None
        self.gmm_vtln_args = None
        self.gmm_sat_args = None
        self.gmm_vtln_sat_args = None

        self.cart_questions = None
        self.lda_matrices = {}
        self.cart_trees = {}

        self.vtln_files = {}

        self.default_align_keep_values = {
            "default": 5,
            "selected": gs.JOB_DEFAULT_KEEP_VALUE,
        }

    # -------------------- Setup --------------------
    def init_system(
        self,
        hybrid_init_args: RasrInitArgs,
        gmm_monophone_args: GmmMonophoneArgs,
        gmm_triphone_args: Optional[GmmTriphoneArgs],
        gmm_vtln_args: Optional[GmmVtlnArgs],
        gmm_sat_args: Optional[GmmSatArgs],
        gmm_vtln_sat_args: Optional[GmmVtlnSatArgs],
        train_data: Dict[str, RasrDataInput],
        dev_data: Dict[str, RasrDataInput],
        test_data: Dict[str, RasrDataInput],
    ):
        self.hybrid_init_args = hybrid_init_args
        self.gmm_monophone_args = gmm_monophone_args
        self.gmm_triphone_args = gmm_triphone_args
        self.gmm_vtln_args = gmm_vtln_args
        self.gmm_sat_args = gmm_sat_args
        self.gmm_vtln_sat_args = gmm_vtln_sat_args

        self._init_am(**self.hybrid_init_args.am_args)

        self._assert_corpus_name_unique(train_data, dev_data, test_data)

        for name, v in sorted(train_data.items()):
            add_lm = True if v.lm is not None else False
            self.add_corpus(name, data=v, add_lm=add_lm)
            self.train_corpora.append(name)

        for name, v in sorted(dev_data.items()):
            self.add_corpus(name, data=v, add_lm=True)
            self.dev_corpora.append(name)

        for name, v in sorted(test_data.items()):
            self.add_corpus(name, data=v, add_lm=True)
            self.test_corpora.append(name)

        self.cart_questions = self.gmm_triphone_args.cart_questions if self.gmm_triphone_args else None

    # -------------------- Mono Training --------------------

    @tk.block()
    def monophone_training(
        self,
        name: str,
        corpus: str,
        linear_alignment_args: dict,
        feature_energy_flow: str,
        feature_flow: str,
        align_iter: int,
        splits: int,
        accs_per_split: int,
        align_keep_values: Optional[dict] = None,
        **kwargs,
    ):
        self.linear_alignment(
            name,
            corpus,
            feature_energy_flow,
            prefix=f"{corpus}_",
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
            corpus=corpus,
            sequence=action_sequence,
            flow=feature_flow,
            initial_mixtures=meta.select_element(
                self.mixtures, corpus, "linear_alignment_{}".format(name)
            ),
            align_keep_values=akv,
            **kwargs,
        )
        self.jobs[corpus]["train_{}".format(name)].selected_alignment_jobs[
            -1
        ].add_alias("train/{}_{}_align_last".format(corpus, name))

        self.jobs[corpus]["train_{}".format(name)].selected_mixture_jobs[-1].add_alias(
            "train/{}_{}_mix_last".format(corpus, name)
        )
        tk.register_output(
            "train/{}_{}_align_bundle_last".format(corpus, name),
            self.jobs[corpus]["train_{}".format(name)]
            .selected_alignment_jobs[-1]
            .out_alignment_bundle,
        )
        tk.register_output(
            "train/{}_{}_mix_last".format(corpus, name),
            self.jobs[corpus]["train_{}".format(name)]
            .selected_mixture_jobs[-1]
            .out_mixtures,
        )

    # -------------------- CaRT and LDA --------------------

    def cart_and_lda(
        self,
        name: str,
        corpus: str,
        initial_flow: str,
        context_flow: str,
        context_size: int,
        alignment: str,
        num_dim: int,
        num_iter: int,
        eigenvalue_args: dict,
        generalized_eigenvalue_args: dict,
        **kwargs,
    ):
        for f in self.feature_flows.values():
            f["{}+context".format(context_flow)] = lda.add_context_flow(
                feature_net=f[context_flow],
                max_size=context_size,
                right=int(context_size / 2.0),
            )

        cart_lda = meta.CartAndLDA(
            original_crp=self.crp[corpus],
            initial_flow=self.feature_flows[corpus][initial_flow],
            context_flow=self.feature_flows[corpus]["{}+context".format(context_flow)],
            alignment=meta.select_element(self.alignments, corpus, alignment),
            questions=self.cart_questions,
            num_dim=num_dim,
            num_iter=num_iter,
            eigenvalue_args=eigenvalue_args,
            generalized_eigenvalue_args=generalized_eigenvalue_args,
        )
        self.jobs[corpus]["cart_and_lda_{}_{}".format(corpus, name)] = cart_lda
        self.lda_matrices["{}_{}".format(corpus, name)] = cart_lda.last_lda_matrix
        self.cart_trees["{}_{}".format(corpus, name)] = cart_lda.last_cart_tree
        tk.register_output(
            "{}_{}_last_num_cart_labels".format(corpus, name),
            cart_lda.last_num_cart_labels,
        )
        tk.register_output(
            "{}_{}.tree.xml.gz".format(corpus, name), cart_lda.last_cart_tree
        )

        for f in self.feature_flows.values():
            f["{}+context+lda".format(context_flow)] = features.add_linear_transform(
                f["{}+context".format(context_flow)], cart_lda.last_lda_matrix
            )

        for crp in self.crp.values():
            crp.acoustic_model_config.state_tying.type = "cart"
            crp.acoustic_model_config.state_tying.file = cart_lda.last_cart_tree

        state_tying_job = allophones.DumpStateTyingJob(self.crp[corpus])
        tk.register_output(
            "{}_{}_state_tying".format(corpus, name), state_tying_job.out_state_tying
        )

    # -------------------- Tri Training --------------------

    @tk.block()
    def triphone_training(
        self,
        name: str,
        corpus: str,
        feature_flow: str,
        initial_alignment: str,
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
            corpus=corpus,
            sequence=action_sequence,
            flow=feature_flow,
            initial_alignment=meta.select_element(
                self.alignments, corpus, initial_alignment
            ),
            align_keep_values=akv,
            **kwargs,
        )
        self.jobs[corpus]["train_{}".format(name)].selected_alignment_jobs[
            -1
        ].add_alias("train/{}_{}_align_last".format(corpus, name))

        self.jobs[corpus]["train_{}".format(name)].selected_mixture_jobs[-1].add_alias(
            "train/{}_{}_mix_last".format(corpus, name)
        )
        tk.register_output(
            "train/{}_{}_align_bundle_last".format(corpus, name),
            self.jobs[corpus]["train_{}".format(name)]
            .selected_alignment_jobs[-1]
            .out_alignment_bundle,
        )
        tk.register_output(
            "train/{}_{}_mix_last".format(corpus, name),
            self.jobs[corpus]["train_{}".format(name)]
            .selected_mixture_jobs[-1]
            .out_mixtures,
        )

    # -------------------- Vocal Tract Length Normalization --------------------

    def vtln_feature_flow(
        self,
        name: str,
        corpus: str,
        base_flow: str,
        context_size: Optional[int] = None,
        lda_matrix: Optional[str] = None,
    ):
        flow = self.feature_flows[corpus][base_flow]
        if context_size is not None:
            flow = lda.add_context_flow(
                feature_net=flow,
                max_size=context_size,
                right=int(context_size / 2.0),
            )
        if lda_matrix is not None:
            flow = features.add_linear_transform(flow, self.lda_matrices[lda_matrix])
        self.feature_flows[corpus][name] = flow

    @tk.block()
    def vtln_warping_mixtures(
        self,
        name: str,
        corpus: str,
        feature_flow: str,
        feature_scorer: str,
        alignment: str,
        splits: int,
        accs_per_split: int,
    ):
        feature_flow = self.feature_flows[corpus][feature_flow]
        warp = vtln.ScoreFeaturesWithWarpingFactorsJob(
            crp=self.crp[corpus],
            feature_flow=feature_flow,
            feature_scorer=meta.select_element(
                self.feature_scorers, corpus, feature_scorer
            ),
            alignment=meta.select_element(self.alignments, corpus, alignment),
        )
        warp.rqmt = {"time": 24, "cpu": 1, "mem": 2}
        self.jobs[corpus]["vtln_warping_map_%s" % name] = warp

        seq = meta.TrainWarpingFactorsSequence(
            self.crp[corpus],
            None,
            feature_flow,
            warp.warping_map,
            warp.alphas_file,
            ["accumulate"] + meta.split_and_accumulate_sequence(splits, accs_per_split),
        )
        self.mixtures[corpus]["vtln_warping_mix_%s" % name] = seq.selected_mixtures
        self.vtln_files[name + "_alphas_file"] = warp.alphas_file
        self.vtln_files[name + "_warping_map"] = warp.warping_map
        self.vtln_files[name + "_mixtures"] = seq.selected_mixtures

    @tk.block()
    def extract_vtln_features(
        self,
        name: str,
        train_corpus: str,
        eval_corpus: str,
        raw_feature_flow: str,
        vtln_files: str,
        **kwargs,
    ):
        self.vtln_features(
            name=name,
            corpus=train_corpus,
            raw_feature_flow=self.feature_flows[eval_corpus][raw_feature_flow],
            warping_map=self.vtln_files[vtln_files + "_warping_map"],
            **kwargs,
        )
        self.feature_flows[eval_corpus][
            raw_feature_flow + "+vtln"
        ] = vtln.recognized_warping_factor_flow(
            self.feature_flows[eval_corpus][raw_feature_flow],
            self.vtln_files[vtln_files + "_alphas_file"],
            self.vtln_files[vtln_files + "_mixtures"][-1],
        )

    @tk.block()
    def vtln_training(
        self,
        name: str,
        corpus: str,
        initial_alignment: str,
        feature_flow: str,
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
            corpus=corpus,
            sequence=action_sequence,
            flow=feature_flow,
            initial_alignment=self.alignments[corpus][initial_alignment][-1],
            align_keep_values=akv,
            **kwargs,
        )
        self.jobs[corpus]["train_{}".format(name)].selected_alignment_jobs[
            -1
        ].add_alias("train/{}_{}_align_last".format(corpus, name))

        self.jobs[corpus]["train_{}".format(name)].selected_mixture_jobs[-1].add_alias(
            "train/{}_{}_mix_last".format(corpus, name)
        )
        tk.register_output(
            "train/{}_{}_align_bundle_last".format(corpus, name),
            self.jobs[corpus]["train_{}".format(name)]
            .selected_alignment_jobs[-1]
            .out_alignment_bundle,
        )
        tk.register_output(
            "train/{}_{}_mix_last".format(corpus, name),
            self.jobs[corpus]["train_{}".format(name)]
            .selected_mixture_jobs[-1]
            .out_mixtures,
        )

    # -------------------- Speaker Adaptive Training  --------------------

    def estimate_cmllr(
        self,
        name: str,
        corpus: str,
        feature_cache: rasr.FlagDependentFlowAttribute,
        feature_flow: str,
        cache_regex: str,
        alignment: rasr.FlagDependentFlowAttribute,
        mixtures: rasr.FlagDependentFlowAttribute,
        overlay: Optional[str] = None,
    ):
        speaker_seg = corpus_recipes.SegmentCorpusBySpeakerJob(
            self.corpora[corpus].corpus_file
        )
        old_segment_path = self.crp[corpus].segment_path.hidden_paths

        if isinstance(alignment, util.MultiOutputPath):
            alignment = alignment.hidden_paths
        elif isinstance(alignment, rasr.FlagDependentFlowAttribute):
            alignment = alignment.alternatives["task_dependent"].hidden_paths
        else:
            raise Exception("Do not like the type")

        mapped_alignment = rasr.MapSegmentsWithBundlesJob(
            old_segments=old_segment_path,
            cluster_map=speaker_seg.cluster_map_file,
            files=alignment,
            filename="cluster.$(TASK).bundle",
        )
        mapped_features = rasr.MapSegmentsWithBundlesJob(
            old_segments=old_segment_path,
            cluster_map=speaker_seg.cluster_map_file,
            files=feature_cache.hidden_paths,
            filename="cluster.$(TASK).bundle",
        )
        new_segments = rasr.ClusterMapToSegmentListJob(speaker_seg.cluster_map_file)

        overlay = "%s_cmllr_%s" % (corpus, name) if overlay is None else overlay
        self.add_overlay(corpus, overlay)
        self.crp[overlay].segment_path = new_segments.segment_path
        self.replace_named_flow_attr(
            overlay, cache_regex, "cache", mapped_features.bundle_path
        )

        cmllr = sat.EstimateCMLLRJob(
            crp=self.crp[overlay],
            feature_flow=self.feature_flows[overlay][feature_flow],
            mixtures=mixtures,
            alignment=mapped_alignment.bundle_path,
            cluster_map=speaker_seg.cluster_map_file,
            num_clusters=speaker_seg.num_speakers,
        )
        cmllr.rqmt = {
            "time": max(
                self.crp[overlay].corpus_duration
                / (0.2 * self.crp[overlay].concurrent),
                1.0,
            ),
            "cpu": 6,
            "mem": 16,
        }
        self.feature_flows[corpus]["%s+cmllr" % feature_flow] = sat.add_cmllr_transform(
            self.feature_flows[corpus][feature_flow],
            speaker_seg.cluster_map_file,
            cmllr.transforms,
        )

        self.jobs[corpus]["segment_corpus_by_speaker"] = speaker_seg
        self.jobs[overlay]["mapped_alignment"] = mapped_alignment
        self.jobs[overlay]["mapped_features"] = mapped_features
        self.jobs[overlay]["new_segments"] = new_segments
        self.jobs[overlay]["cmllr"] = cmllr

    @tk.block()
    def sat_training(
        self,
        name: str,
        corpus: str,
        feature_cache: str,
        feature_flow: str,
        cache_regex: str,
        alignment: str,
        mixtures: str,
        splits: int,
        accs_per_split: int,
        align_keep_values: Optional[dict] = None,
        **kwargs,
    ):
        self.estimate_cmllr(
            name=name,
            corpus=corpus,
            feature_cache=meta.select_element(
                self.feature_caches, corpus, feature_cache
            ),
            feature_flow=feature_flow,
            cache_regex=cache_regex,
            alignment=meta.select_element(self.alignments, corpus, alignment),
            mixtures=meta.select_element(self.mixtures, corpus, mixtures),
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
            corpus=corpus,
            sequence=action_sequence,
            flow="%s+cmllr" % feature_flow,
            initial_alignment=meta.select_element(self.alignments, corpus, alignment),
            align_keep_values=akv,
            **kwargs,
        )
        self.jobs[corpus]["train_{}".format(name)].selected_alignment_jobs[
            -1
        ].add_alias("train/{}_{}_align_last".format(corpus, name))

        self.jobs[corpus]["train_{}".format(name)].selected_mixture_jobs[-1].add_alias(
            "train/{}_{}_mix_last".format(corpus, name)
        )
        tk.register_output(
            "train/{}_{}_align_bundle_last".format(corpus, name),
            self.jobs[corpus]["train_{}".format(name)]
            .selected_alignment_jobs[-1]
            .out_alignment_bundle,
        )
        tk.register_output(
            "train/{}_{}_mix_last".format(corpus, name),
            self.jobs[corpus]["train_{}".format(name)]
            .selected_mixture_jobs[-1]
            .out_mixtures,
        )

    # -------------------- recognition  --------------------

    def recognition(
        self,
        name: str,
        iters: List[int],
        lm_scales: List[float],
        feature_scorer: Tuple[str, str],
        optimize_am_lm_scale: bool,
        # parameters just for passing through
        corpus: str,
        feature_flow: str,
        pronunciation_scale: float,
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
        :param feature_scorer: (training_corpus_name, training_name)
        :param optimize_am_lm_scale: will optimize the lm-scale and re-run recognition with the optimal value
        :param kwargs: see meta.System.recog and meta.System.recog_and_optimize
        :return:
        """
        assert "lm_scale" not in kwargs, "please use lm_scales for GmmSystem.recognition()"
        with tk.block(f"{name}_recognition"):
            recog_func = self.recog_and_optimize if optimize_am_lm_scale else self.recog

            for it, l in itertools.product(iters, lm_scales):
                recog_func(
                    name=f"iter{it:02d}-lm{l}",
                    prefix=f"recognition/{name}/",
                    corpus=corpus,
                    flow=feature_flow,
                    feature_scorer=list(feature_scorer) + [it - 1],
                    pronunciation_scale=pronunciation_scale,
                    lm_scale=l,
                    search_parameters=search_parameters,
                    rtf=rtf,
                    mem=mem,
                    parallelize_conversion=parallelize_conversion,
                    lattice_to_ctm_kwargs=lattice_to_ctm_kwargs,
                    **kwargs,
                )

    # -------------------- run setup  --------------------

    def run(self, steps: Union[List, Tuple] = ("all",)):
        """
        run setup

        :param steps:
        :return:
        """
        assert len(steps) > 0
        if len(steps) == 1 and steps[0] == "all":
            steps = ["extract", "mono", "cart", "tri", "vtln", "sat", "vtln+sat"]

        if "init" in steps:
            print(
                "init needs to be run manually. provide: gmm_args, {train,dev,test}_inputs"
            )
            sys.exit(-1)

        for all_c in self.train_corpora + self.dev_corpora + self.test_corpora:
            self.costa(all_c, prefix="costa/", **self.hybrid_init_args.costa_args)

        for trn_c in self.train_corpora:
            self.store_allophones(trn_c)

        for eval_c in self.dev_corpora + self.test_corpora:
            self.create_stm_from_corpus(eval_c)
            self.set_sclite_scorer(eval_c)

        if "extract" in steps:
            self.extract_features(
                feat_args=self.hybrid_init_args.feature_extraction_args
            )

        # ---------- Monophone ----------
        if "mono" in steps:
            for trn_c in self.train_corpora:
                self.monophone_training(
                    "mono",
                    trn_c,
                    self.gmm_monophone_args.linear_alignment_args,
                    **self.gmm_monophone_args.monophone_training_args,
                )

                for dev_c in self.dev_corpora:
                    feature_scorer = (trn_c, "train_mono")
                    self.recognition(
                        f"mono-{trn_c}-{dev_c}",
                        corpus=dev_c,
                        feature_scorer=feature_scorer,
                        **self.gmm_monophone_args.monophone_recognition_args,
                    )

                for tst_c in self.test_corpora:
                    pass

        # ---------- CaRT ----------
        if "cart" in steps:
            for c in self.train_corpora:
                self.cart_and_lda(
                    "mono",
                    c,
                    alignment="train_mono",
                    **self.gmm_triphone_args.cart_lda_args,
                )

        # ---------- Triphone ----------
        if "tri" in steps:
            for c in self.train_corpora:
                self.triphone_training(
                    "tri",
                    c,
                    initial_alignment="train_mono",
                    **self.gmm_triphone_args.triphone_training_args,
                )

            for c in self.dev_corpora:
                # recog and optimize
                pass

            for c in self.test_corpora:
                pass

            # ---------- SDM Tri ----------
            for c in self.train_corpora:
                self.single_density_mixtures(
                    "sdm.tri",
                    c,
                    alignment="train_tri",
                    **self.gmm_triphone_args.sdm_tri_args,
                )

        # ---------- VTLN ----------
        if "vtln" in steps:
            for c in self.dev_corpora + self.test_corpora:
                self.vtln_feature_flow(
                    "uncached_mfcc+context+lda",
                    c,
                    lda_matrix="batch1+2.train_mono",
                    **self.gmm_vtln_args.vtln_training_args["feature_flow"],
                )

            for c in self.train_corpora:
                self.vtln_feature_flow(
                    "uncached_mfcc+context+lda",
                    c,
                    lda_matrix="batch1+2.train_mono",
                    **self.gmm_vtln_args.vtln_training_args["feature_flow"],
                )

                self.vtln_warping_mixtures(
                    "vtln",
                    c,
                    feature_scorer="estimate_mixtures_sdm.tri",
                    **self.gmm_vtln_args.vtln_training_args["warp_mix"],
                )

                for cc in self.dev_corpora + self.test_corpora:
                    self.extract_vtln_features(
                        "mfcc+context+lda",
                        train_corpus=c,
                        eval_corpus=cc,
                        raw_feature_flow="uncached_mfcc+context+lda",
                        vtln_files="vtln",
                    )

                self.vtln_training(
                    "vtln",
                    c,
                    initial_alignment="train_tri",
                    **self.gmm_vtln_args.vtln_training_args["train"],
                )

            for c in self.test_corpora:
                pass

            for c in self.test_corpora:
                pass

            # ---------- SDM VTLN ----------
            for c in self.train_corpora:
                self.single_density_mixtures(
                    "sdm.vtln",
                    c,
                    alignment="train_vtln",
                    **self.gmm_vtln_args.sdm_vtln_args,
                )

        # ---------- SAT ----------
        if "sat" in steps:
            for c in self.train_corpora:
                self.sat_training(
                    "sat",
                    c,
                    mixtures="estimate_mixtures_sdm.tri",
                    alignment="train_tri",
                    **self.gmm_sat_args.sat_training_args,
                )

            for c in self.dev_corpora:
                # recog and optimize
                pass

            for c in self.test_corpora:
                pass

            # ---------- SDM Sat ----------
            for c in self.train_corpora:
                self.single_density_mixtures(
                    "sdm.sat",
                    c,
                    alignment="train_sat",
                    **self.gmm_sat_args.sdm_sat_args,
                )

        # ---------- VTLN+SAT ----------
        if "vtln+sat" in steps:
            for c in self.train_corpora:
                self.single_density_mixtures(
                    "sdm.vtln",
                    c,
                    alignment="train_vtln",
                    **self.gmm_vtln_sat_args.sdm_vtln_args,
                )
                self.sat_training(
                    "vtln_sat",
                    c,
                    mixtures="estimate_mixtures_sdm.vtln",
                    alignment="train_vtln",
                    **self.gmm_vtln_sat_args.vtln_sat_training_args,
                )

            for c in self.dev_corpora:
                # recog and optimize
                pass

            for c in self.test_corpora:
                pass

            # ---------- SDM VTLN+SAT ----------
            for c in self.train_corpora:
                self.single_density_mixtures(
                    "sdm.sat_vtln",
                    c,
                    alignment="train_vtln_sat",
                    **self.gmm_triphone_args.sdm_tri_args,
                )
