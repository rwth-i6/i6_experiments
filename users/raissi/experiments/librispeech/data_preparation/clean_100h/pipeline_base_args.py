__all__ = [
    "get_final_output",
]

from typing import Dict, Optional, Union

# -------------------- Sisyphus --------------------

from sisyphus import tk

# -------------------- Recipes --------------------

import i6_core.features as features
import i6_core.rasr as rasr

import i6_experiments.common.datasets.librispeech as lbs_dataset
import i6_experiments.common.setups.rasr.util as rasr_util

# -------------------- helpers --------------------

from i6_experiments.users.raissi.setups.common.data.pipeline_helpers import InputKey


def get_final_output(name=InputKey.BASE):
    output_args = rasr_util.OutputArgs(name)

    output_args.define_corpus_type("train-clean-100", "train")
    output_args.define_corpus_type("dev-clean", "dev")
    output_args.define_corpus_type("dev-other", "dev")
    output_args.define_corpus_type("dev-other_dev-clean", "cv")
    output_args.define_corpus_type("test-clean", "test")
    output_args.define_corpus_type("test-other", "test")

    output_args.add_feature_to_extract("gt")

    return output_args
