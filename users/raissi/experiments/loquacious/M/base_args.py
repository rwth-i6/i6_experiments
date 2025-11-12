__all__ = [
    "get_corpus_data_inputs",
    "get_init_args",
    "get_final_output",
    "DatasetSize"
]

from enum import Enum
from typing import Dict, Optional

# -------------------- Sisyphus --------------------

from sisyphus import tk

# -------------------- Recipes --------------------
import i6_experiments.common.setups.rasr.util as rasr_util
from i6_experiments.users.raissi.setups.common.data.pipeline_helpers import InputKey

from i6_experiments.users.raissi.experiments.loquacious.common.base_args import (
    DEV_KEYS, TEST_KEYS
)


def get_final_output(name=InputKey.BASE):
    output_args = rasr_util.OutputArgs(name)
    output_args.define_corpus_type("train.medium", "train.medium")
    for k in DEV_KEYS+TEST_KEYS:
        output_args.define_corpus_type(k, k)
    output_args.add_feature_to_extract("fb")

    return output_args
