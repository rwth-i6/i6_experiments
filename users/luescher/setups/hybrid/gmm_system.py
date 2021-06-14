__all__ = ["GmmSystem"]

import itertools
import sys
from typing import Dict

# -------------------- Sisyphus --------------------

from sisyphus import *
import sisyphus.toolkit as tk
import sisyphus.global_settings as gs

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

from .util import GmmDataInput, GmmPipelineArgs

# -------------------- Init --------------------

Path = tk.setup_path(__package__)


# -------------------- System --------------------


class GmmSystem(meta.System):
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

        self.crp["base"].python_home = gs.RASR_PYTHON_HOME
        self.crp["base"].python_program_name = gs.RASR_PYTHON_EXE

        self.gmm_args = None

        self.train_corpora = []
        self.dev_corpora = []
        self.test_corpora = []

        self.corpora = {}
        self.concurrent = {}

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
        gmm_args: GmmPipelineArgs,
        train_data: Dict[str, GmmDataInput],
        dev_data: Dict[str, GmmDataInput],
        test_data: Dict[str, GmmDataInput],
    ):
        """
        :param gmm_args: parameters for the different Gmm-HMM steps
        :param train_data: dict(str: GmmInput)
        :param dev_data: dict(str: GmmInput)
        :param test_data: dict(str: GmmInput)
        :return:
        """
        self.gmm_args = gmm_args
        self._init_am(**gmm_args.am_args)
        for name, v in sorted(train_data.items()):
            self.corpora[name] = v.corpus_object
            self.concurrent[name] = v.concurrent
            self._init_corpora(name)
            self._init_lexica(name, **v.lexicon)
            self.train_corpora.append(name)

        for name, v in sorted(dev_data.items()):
            self.corpora[name] = v.corpus_object
            self.concurrent[name] = v.concurrent
            self._init_corpora(name)
            self._init_lexica(name, **v.lexicon)
            self._init_lm(name, **v.lm)
            self.dev_corpora.append(name)

        for name, v in sorted(test_data.items()):
            self.corpora[name] = v.corpus_object
            self.concurrent[name] = v.concurrent
            self._init_corpora(name)
            self._init_lexica(name, **v.lexicon)
            self._init_lm(name, **v.lm)
            self.test_corpora.append(name)
        self.cart_questions = gmm_args.cart_questions

    @tk.block()
    def _init_am(self, **kwargs):
        self.crp["base"].acoustic_model_config = am.acoustic_model_config(**kwargs)

    @tk.block()
    def _init_corpora(self, name):
        segm_corpus_job = corpus_recipes.SegmentCorpusJob(
            self.corpora[name].corpus_file, self.concurrent[name]
        )
        self.set_corpus(
            name=name,
            corpus=self.corpora[name],
            concurrent=self.concurrent[name],
            segment_path=segm_corpus_job.segment_path,
        )
        self.jobs[name]["segment_corpus"] = segm_corpus_job

    @tk.block()
    def _init_lm(self, name, file, type, scale, **kwargs):
        self.crp[name].language_model_config = rasr.RasrConfig()
        self.crp[name].language_model_config.type = type
        self.crp[name].language_model_config.file = file
        self.crp[name].language_model_config.scale = scale

    @tk.block()
    def _init_lexica(self, name, file, normalize_pronunciation, **kwargs):
        self.crp[name].lexicon_config = rasr.RasrConfig()
        self.crp[name].lexicon_config.file = file
        self.crp[name].lexicon_config.normalize_pronunciation = normalize_pronunciation

    def allow_zero_weights(self, name: str = "base"):
        if self.crp[name].acoustic_model_post_config is None:
            self.crp[name].acoustic_model_post_config = rasr.RasrConfig()
        self.crp[name].acoustic_model_post_config.mixture_set.allow_zero_weights = True
        self.crp[
            name
        ].acoustic_model_post_config.old_mixture_set.allow_zero_weights = True

    def _add_features(self, name, corpus, feature_job, prefix=""):
        self.jobs[corpus][name] = feature_job
        feature_job.add_alias(f"{prefix}{corpus}_{name}_features")
        self.feature_caches[corpus][name] = feature_job.out_feature_path[name]
        self.feature_bundles[corpus][name] = feature_job.out_feature_bundle[name]
        feature_path = rasr.FlagDependentFlowAttribute(
            "cache_mode",
            {
                "task_dependent": self.feature_caches[corpus][name],
                "bundle": self.feature_bundles[corpus][name],
            },
        )
        self.feature_flows[corpus][name] = features.basic_cache_flow(feature_path)
        self.feature_flows[corpus][f"uncached_{name}"] = feature_job.feature_flow

    @tk.block()
    def extract_features(self, feat_args: dict, **kwargs):
        corpus_list = self.train_corpora + self.dev_corpora + self.test_corpora

        if "mfcc" in feat_args.keys():
            for c in corpus_list:
                self.mfcc_features(c, **feat_args["mfcc"])
                self.energy_features(c, **feat_args["energy"])
            for t in self.train_corpora:
                self.add_energy_to_features(t, "mfcc+deriv")

        if "gt" in feat_args.keys():
            for c in corpus_list:
                self.gt_features(c, **feat_args["gt"])
                self.energy_features(c, **feat_args["energy"])
            for t in self.train_corpora:
                self.add_energy_to_features(t, "gt")

        if "fb" in feat_args.keys():
            for c in corpus_list:
                self.fb_features(c, **feat_args["fb"])

        unknown_features = set(feat_args.keys()).difference(
            {"mfcc", "gt", "fb", "energy"}
        )
        if len(unknown_features) > 0:
            raise ValueError("Invalid features: {}".format(unknown_features))

    # -------------------- Mono Training --------------------

    @tk.block()
    def monophone_training(
        self,
        name,
        corpus,
        linear_alignment_args,
        feature_energy_flow,
        feature_flow,
        align_iter,
        splits,
        accs_per_split,
        align_keep_values=None,
        **kwargs,
    ):
        self.linear_alignment(
            name, corpus, feature_energy_flow, **linear_alignment_args
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
        ].add_alias("train_{}_alnt_last".format(name))
        self.jobs[corpus]["train_{}".format(name)].selected_mixture_jobs[-1].add_alias(
            "train_{}_mix_last".format(name)
        )
        tk.register_output(
            "train_{}_{}_alnt_bundle_last".format(corpus, name),
            self.jobs[corpus]["train_{}".format(name)]
            .selected_alignment_jobs[-1]
            .alignment_bundle,
        )
        tk.register_output(
            "train_{}_{}_mix_last".format(corpus, name),
            self.jobs[corpus]["train_{}".format(name)]
            .selected_mixture_jobs[-1]
            .mixtures,
        )

    # -------------------- CaRT and LDA --------------------

    def cart_and_lda(
        self,
        name,
        corpus,
        initial_flow,
        context_flow,
        context_size,
        alignment,
        num_dim,
        num_iter,
        eigenvalue_args,
        generalized_eigenvalue_args,
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
            "{}_{}_state_tying".format(corpus, name), state_tying_job.state_tying
        )

    # -------------------- Tri Training --------------------

    @tk.block()
    def triphone_training(
        self,
        name,
        corpus,
        feature_flow,
        initial_alignment,
        splits,
        accs_per_split,
        align_keep_values=None,
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
        ].add_alias("train_{}_alnt_last".format(name))
        self.jobs[corpus]["train_{}".format(name)].selected_mixture_jobs[-1].add_alias(
            "train_{}_mix_last".format(name)
        )
        tk.register_output(
            "train_{}_{}_alnt_bundle_last".format(corpus, name),
            self.jobs[corpus]["train_{}".format(name)]
            .selected_alignment_jobs[-1]
            .alignment_bundle,
        )
        tk.register_output(
            "train_{}_{}_mix_last".format(corpus, name),
            self.jobs[corpus]["train_{}".format(name)]
            .selected_mixture_jobs[-1]
            .mixtures,
        )

    # -------------------- Single Density Mixtures --------------------

    def single_density_mixtures(self, name, corpus, feature_flow, alignment):
        self.estimate_mixtures(
            name=name,
            corpus=corpus,
            flow=feature_flow,
            alignment=meta.select_element(self.alignments, corpus, alignment),
            split_first=False,
        )

    # -------------------- Vocal Tract Length Normalization --------------------

    def vtln_feature_flow(
        self, name, corpora, base_flow, context_size=None, lda_matrix=None
    ):
        for corpus in corpora:
            flow = self.feature_flows[corpus][base_flow]
            if context_size is not None:
                flow = lda.add_context_flow(
                    feature_net=flow,
                    max_size=context_size,
                    right=int(context_size / 2.0),
                )
            if lda_matrix is not None:
                flow = features.add_linear_transform(
                    flow, self.lda_matrices[lda_matrix]
                )
            self.feature_flows[corpus][name] = flow

    @tk.block()
    def vtln_warping_mixtures(
        self,
        name,
        corpus,
        feature_flow,
        feature_scorer,
        alignment,
        splits,
        accs_per_split,
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
        self, name, train_corpus, eval_corpus, raw_feature_flow, vtln_files, **kwargs
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
        name,
        corpus,
        initial_alignment,
        feature_flow,
        splits,
        accs_per_split,
        align_keep_values=None,
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
        ].add_alias("train_{}_alnt_last".format(name))
        self.jobs[corpus]["train_{}".format(name)].selected_mixture_jobs[-1].add_alias(
            "train_{}_mix_last".format(name)
        )
        tk.register_output(
            "train_{}_{}_alnt_bundle_last".format(corpus, name),
            self.jobs[corpus]["train_{}".format(name)]
            .selected_alignment_jobs[-1]
            .alignment_bundle,
        )
        tk.register_output(
            "train_{}_{}_mix_last".format(corpus, name),
            self.jobs[corpus]["train_{}".format(name)]
            .selected_mixture_jobs[-1]
            .mixtures,
        )

    # -------------------- Speaker Adaptive Training  --------------------

    def estimate_cmllr(
        self,
        name,
        corpus,
        feature_cache,
        feature_flow,
        cache_regex,
        alignment,
        mixtures,
        overlay=None,
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
        name,
        corpus,
        feature_cache,
        feature_flow,
        cache_regex,
        alignment,
        mixtures,
        splits,
        accs_per_split,
        align_keep_values=None,
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
        ].add_alias("train_{}_alnt_last".format(name))
        self.jobs[corpus]["train_{}".format(name)].selected_mixture_jobs[-1].add_alias(
            "train_{}_mix_last".format(name)
        )
        tk.register_output(
            "train_{}_alnt_bundle_last".format(name),
            self.jobs[corpus]["train_{}".format(name)]
            .selected_alignment_jobs[-1]
            .alignment_bundle,
        )
        tk.register_output(
            "train_{}_mix_last".format(name),
            self.jobs[corpus]["train_{}".format(name)]
            .selected_mixture_jobs[-1]
            .mixtures,
        )

    # -------------------- recognition  --------------------

    def recognition(
        self,
        name,
        iters,
        corpus,
        feature_flow,
        feature_scorer,
        pronunciation_scale,
        lm_scale,
        search_params,
        rtf,
        mem,
        parallelize_conversion,
        lattice_to_ctm_kwargs,
        **kwargs,
    ):
        with tk.block(f"{name}_recognition"):
            optimize_am_lm_scale = self.gmm_args.monophone_recognition_args.pop(
                "optimize_am_lm_scale", False
            )
            recog_func = self.recog_and_optimize if optimize_am_lm_scale else self.recog

            for it in iters:
                recog_func(
                    name=f"{name}-{it:02d}",
                    prefix=f"recog_{corpus}_{name}/",
                    corpus=corpus,
                    flow=feature_flow,
                    feature_scorer=list(feature_scorer) + [it - 1],
                    pronunciation_scale=pronunciation_scale,
                    lm_scale=lm_scale,
                    search_parameters=search_params,
                    rtf=rtf,
                    mem=mem,
                    parallelize_conversion=parallelize_conversion,
                    lattice_to_ctm_kwargs=lattice_to_ctm_kwargs,
                    **kwargs,
                )

    # -------------------- run setup  --------------------

    def run(self, steps=tuple("all")):
        assert len(steps) > 0
        if len(steps) == 1 and steps[0] == "all":
            steps = ["extract", "mono", "cart", "tri", "vtln", "sat", "vtln+sat"]

        if "init" in steps:
            print(
                "init needs to be run manually. provide: gmm_args, {train,dev,test}_inputs"
            )
            sys.exit(-1)

        if "extract" in steps:
            self.extract_features(feat_args=self.gmm_args.feature_extraction_args)

        # ---------- Monophone ----------
        if "mono" in steps:
            for trn_c in self.train_corpora:
                self.monophone_training(
                    "mono",
                    trn_c,
                    self.gmm_args.linear_alignment_args,
                    **self.gmm_args.monophone_training_args,
                )

                for dev_c in self.dev_corpora:
                    pass

                for tst_c in self.test_corpora:
                    pass

        # ---------- CaRT ----------
        if "cart" in steps:
            for c in self.train_corpora:
                self.cart_and_lda(
                    "mono", c, alignment="train_mono", **self.gmm_args.cart_lda_args
                )

        # ---------- Triphone ----------
        if "tri" in steps:
            for c in self.train_corpora:
                self.triphone_training(
                    "tri",
                    c,
                    initial_alignment="train_mono",
                    **self.gmm_args.triphone_training_args,
                )

            for c in self.dev_corpora:
                # recog and optimize
                pass

            for c in self.test_corpora:
                pass

        # ---------- SDM Tri ----------
        if any(x in steps for x in ["vtln", "sat", "sdm"]):
            for c in self.train_corpora:
                self.single_density_mixtures(
                    "sdm.tri", c, alignment="train_tri", **self.gmm_args.sdm_tri_args
                )

        # ---------- VTLN ----------
        if "vtln" in steps:
            for c in self.train_corpora:
                self.vtln_feature_flow(
                    "uncached_mfcc+context+lda",
                    c,
                    lda_matrix="",
                    **self.gmm_args.vtln_feature_flow_args,
                )
                self.vtln_warping_mixtures(
                    "vtln",
                    c,
                    feature_scorer="estimate_mixtures_sdm.tri",
                    **self.gmm_args.vtln_warping_mixtures_args,
                )
                self.vtln_training(
                    "vtln",
                    c,
                    initial_alignment="train_tri",
                    **self.gmm_args.vtln_training_args,
                )

            for c in self.dev_corpora + self.test_corpora:
                self.extract_vtln_features(
                    self.gmm_args.triphone_training_args["feature_flow"],
                    self.train_corpora[0],
                    c,
                    raw_feature_flow="uncached_mfcc+context+lda",
                    vtln_files="vtln",
                )

            for c in self.test_corpora:
                pass

            for c in self.test_corpora:
                pass

        # ---------- SAT ----------
        if "sat" in steps:
            for c in self.train_corpora:
                self.sat_training(
                    "sat",
                    c,
                    mixtures="estimate_mixtures_sdm.tri",
                    alignment="train_tri",
                    **self.gmm_args.sat_training_args,
                )

            for c in self.dev_corpora:
                # recog and optimize
                pass

            for c in self.test_corpora:
                pass

        # ---------- VTLN+SAT ----------
        if "vtln+sat" in steps:
            for c in self.train_corpora:
                self.single_density_mixtures(
                    "sdm.vtln", c, alignment="train_vtln", **self.gmm_args.sdm_vtln_args
                )
                self.sat_training(
                    "vtln_sat",
                    c,
                    mixtures="estimate_mixtures_sdm.vtln",
                    alignment="train_vtln",
                    **self.gmm_args.vtln_sat_training_args,
                )

            for c in self.dev_corpora:
                # recog and optimize
                pass

            for c in self.test_corpora:
                pass
