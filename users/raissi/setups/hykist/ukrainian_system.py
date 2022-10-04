__all__ = ["UkrainianGMMSystem"]

import copy
import itertools
import sys

from dataclasses import asdict
from IPython import embed
from typing import Dict, List, Optional, Tuple, Union

# -------------------- Sisyphus --------------------

import sisyphus.toolkit as tk
import sisyphus.global_settings as gs

from sisyphus.delayed_ops import DelayedFormat

# -------------------- Recipes --------------------
import i6_core.corpus as corpus_recipe
import i6_core.features as features
import i6_core.mm as mm
import i6_core.rasr as rasr
import i6_core.returnn as returnn
import i6_core.text as text

from i6_core.util import MultiPath, MultiOutputPath


from i6_experiments.common.setups.rasr.nn_system import (
    NnSystem
)

from i6_experiments.common.setups.rasr.gmm_system import (
GmmSystem
)

from i6_experiments.common.setups.rasr.util import (
    OggZipHdfDataInput,
    RasrInitArgs,
    RasrDataInput,
    ReturnnRasrDataInput,
    RasrSteps,
)

from i6_experiments.common.datasets.librispeech.constants import (
num_segments
)


#get_recog_ctx_args*() functions are imported here
from i6_experiments.users.raissi.experiments.librispeech.search.recognition_args import *
from i6_experiments.users.raissi.setups.common.helpers.estimate_povey_like_prior_fh import *


from i6_experiments.users.raissi.returnn.rasr_returnn_bw import (
ReturnnRasrTrainingBWJob
)

from i6_experiments.users.raissi.setups.common.helpers.pipeline_data import (
    ContextEnum,
    ContextMapper,
    LabelInfo,
    PipelineStages,
    SprintFeatureToHdf
)

from i6_experiments.users.raissi.setups.common.helpers.network_architectures import (
    get_graph_from_returnn_config
)

from i6_experiments.users.raissi.setups.common.helpers.train_helpers import (
    get_extra_config_segment_order,
)

from i6_experiments.users.raissi.setups.common.helpers.specaugment_returnn_epilog import (
    get_specaugment_epilog,
)

from i6_experiments.users.raissi.setups.common.helpers.returnn_epilog import (
    get_epilog_code_dense_label,
)

from i6_experiments.users.raissi.setups.common.util.rasr import (
    SystemInput,
)

from i6_experiments.users.raissi.setups.librispeech.util.pipeline_helpers import (
    get_label_info,
    get_alignment_keys,
    get_lexicon_args,
    get_tdp_values,
)

from i6_experiments.users.raissi.setups.librispeech.search.factored_hybrid_search import(
    FHDecoder
)
# -------------------- Init --------------------

Path = tk.setup_path(__package__)

# -------------------- System --------------------

class UkrainianGMMSystem(GmmSystem):
    def run(self, steps: Union[List, Tuple] = ("all",)):
        if "init" in steps:
            print(
                "init needs to be run manually. provide: gmm_args, {train,dev,test}_inputs"
            )
            sys.exit(-1)

        for all_c in self.train_corpora + self.dev_corpora + self.test_corpora:
            costa_args = copy.deepcopy(self.rasr_init_args.costa_args)
            if self.crp[all_c].language_model_config is None:
                costa_args["eval_lm"] = False
            self.costa(all_c, prefix="costa/", **costa_args)
            if costa_args["eval_lm"]:
                self.jobs[all_c]["costa"].update_rqmt("run", {"mem": 24, "time": 24})

        for trn_c in self.train_corpora:
            self.store_allophones(trn_c)

        for eval_c in self.dev_corpora + self.test_corpora:
            self.create_stm_from_corpus(eval_c)
            self.set_sclite_scorer(eval_c)

        if "extract" in steps:
            self.extract_features(
                feat_args=self.rasr_init_args.feature_extraction_args
            )

        # ---------- Monophone ----------
        if "mono" in steps:
            for trn_c in self.train_corpora:
                self.monophone_training(
                    corpus=trn_c,
                    linear_alignment_args=self.monophone_args.linear_alignment_args,
                    **self.monophone_args.training_args,
                )

                for dev_c in self.dev_corpora:
                    feature_scorer = (
                        trn_c,
                        f"train_{self.monophone_args.training_args['name']}",
                    )

                    self.recognition(
                        f"{self.monophone_args.training_args['name']}-{trn_c}-{dev_c}",
                        corpus=dev_c,
                        feature_scorer=feature_scorer,
                        **self.monophone_args.recognition_args,
                    )

                for tst_c in self.test_corpora:
                    pass

        # ---------- CaRT ----------
        if "cart" in steps:
            for trn_c in self.train_corpora:
                self.cart_and_lda(
                    corpus=trn_c,
                    **self.triphone_args.cart_lda_args,
                )

        # ---------- Triphone ----------
        if "tri" in steps:
            for trn_c in self.train_corpora:
                self.triphone_training(
                    corpus=trn_c,
                    **self.triphone_args.training_args,
                )

                for dev_c in self.dev_corpora:
                    feature_scorer = (
                        trn_c,
                        f"train_{self.triphone_args.training_args['name']}",
                    )

                    self.recognition(
                        f"{self.triphone_args.training_args['name']}-{trn_c}-{dev_c}",
                        corpus=dev_c,
                        feature_scorer=feature_scorer,
                        **self.triphone_args.recognition_args,
                    )

                for c in self.test_corpora:
                    pass

                # ---------- SDM Tri ----------
                self.single_density_mixtures(
                    corpus=trn_c,
                    **self.triphone_args.sdm_args,
                )

        # ---------- VTLN ----------
        if "vtln" in steps:
            for trn_c in self.train_corpora:
                self.vtln_feature_flow(
                    train_corpora=trn_c,
                    corpora=[trn_c] + self.dev_corpora + self.test_corpora,
                    **self.vtln_args.training_args["feature_flow"],
                )

                self.vtln_warping_mixtures(
                    corpus=trn_c,
                    feature_flow=self.vtln_args.training_args["feature_flow"]["name"],
                    **self.vtln_args.training_args["warp_mix"],
                )

                self.extract_vtln_features(
                    name=self.triphone_args.training_args["feature_flow"],
                    train_corpus=trn_c,
                    eval_corpus=self.dev_corpora + self.test_corpora,
                    raw_feature_flow=self.vtln_args.training_args["feature_flow"][
                        "name"
                    ],
                    vtln_files=self.vtln_args.training_args["warp_mix"]["name"],
                )

                self.vtln_training(
                    corpus=trn_c,
                    **self.vtln_args.training_args["train"],
                )

                for dev_c in self.dev_corpora:
                    feature_scorer = (
                        trn_c,
                        f"train_{self.vtln_args.training_args['train']['name']}",
                    )

                    self.recognition(
                        f"{self.vtln_args.training_args['train']['name']}-{trn_c}-{dev_c}",
                        corpus=dev_c,
                        feature_scorer=feature_scorer,
                        **self.vtln_args.recognition_args,
                    )

                for tst_c in self.test_corpora:
                    pass

                # ---------- SDM VTLN ----------
                self.single_density_mixtures(
                    corpus=trn_c,
                    **self.vtln_args.sdm_args,
                )

        # ---------- SAT ----------
        if "sat" in steps:
            for trn_c in self.train_corpora:
                self.sat_training(
                    corpus=trn_c,
                    **self.sat_args.training_args,
                )

                for dev_c in self.dev_corpora:
                    pass

                for tst_c in self.test_corpora:
                    pass

                # ---------- SDM Sat ----------
                self.single_density_mixtures(
                    corpus=trn_c,
                    **self.sat_args.sdm_args,
                )

        # ---------- VTLN+SAT ----------
        if "vtln+sat" in steps:
            for trn_c in self.train_corpora:
                self.sat_training(
                    corpus=trn_c,
                    **self.vtln_sat_args.training_args,
                )

                for dev_c in self.dev_corpora:
                    pass

                for tst_c in self.test_corpora:
                    pass

                # ---------- SDM VTLN+SAT ----------
                self.single_density_mixtures(
                    corpus=trn_c,
                    **self.vtln_sat_args.sdm_args,
                )

