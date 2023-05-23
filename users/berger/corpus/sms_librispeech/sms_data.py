from i6_core.returnn.config import CodeWrapper
from i6_experiments.common.datasets.sms_wsj import returnn_datasets
from i6_experiments.common.datasets.sms_wsj.helpers import (
    segment_to_rasr_librispeech_original,
)
from sisyphus import tk
from ..general import SMSHybridSetupData
from .data import get_data_inputs, get_scoring_corpora


json_path = tk.Path(
    "/work/asr3/converse/data/librispeech/sms_librispeech.train_cv.wav.json",
    cached=True,
)

zip_cache_path = "/work/asr3/converse/data/sms_librispeech_original_and_rir_compact_train_cv.zip"
zip_prefix = "/work/asr3/converse/data/"

num_classes = 12001

train_align_hdf = [
    tk.Path(
        f"/work/asr4/berger/dependencies/librispeech/alignments/train-960_cart-12001/data.hdf.{i}",
        cached=True,
    )
    for i in range(200)
]
cv_align_hdf = [
    tk.Path(
        f"/work/asr4/berger/dependencies/librispeech/alignments/train-960_cart-12001/data.hdf.{i}",
        cached=True,
    )
    for i in range(200)
]


def get_sms_data() -> SMSHybridSetupData:
    datasets = {
        key: {
            "class": CodeWrapper("SmsWsjMixtureEarlyAlignmentDataset"),
            "num_outputs": {
                "data": {"dim": 1, "shape": (None, 1)},
                "target_signals": {"dim": 2, "shape": (None, 2)},
                "target_classes": {
                    "dim": num_classes,
                    "shape": (None, 2),
                    "sparse": True,
                },
            },
            "seq_ordering": "default",
            "partition_epoch": {"train": 20, "cv": 1}[key],
            "sms_wsj_kwargs": {
                "dataset_name": f"train_960_{key}",
                "json_path": json_path,
                "zip_cache": zip_cache_path,
                "zip_prefix": zip_prefix,
                "shuffle": {"train": True, "cv": False}[key],
                "hdf_file": {"train": train_align_hdf, "cv": cv_align_hdf}[key],
                "segment_mapping_fn": CodeWrapper(
                    "partial(segment_to_rasr_librispeech_original, prefix='train-other-960')"
                ),
                "pad_label": num_classes - 1,
                "hdf_data_key": "data",
            },
        }
        for key in ["train", "cv"]
    }

    python_prolog = {
        "modules": [
            "import functools",
            "from functools import partial",
            "import json",
            "import numpy as np",
            "import os.path",
            "import re",
            "import tensorflow as tf",
            "import subprocess as sp",
            "import sys",
            "sys.setrecursionlimit(3000)",
            "sys.path.append(os.path.dirname('/u/berger/asr-exps/recipe/i6_core'))",
            "from typing import Dict, List, Tuple, Any, Optional",
            "from returnn.datasets.basic import DatasetSeq",
            "from returnn.datasets.hdf import HDFDataset",
            "from returnn.datasets.map import MapDatasetBase, MapDatasetWrapper",
            "from returnn.log import log as returnn_log",
            "from returnn.util.basic import OptionalNotImplementedError, NumbersDict",
            "from sms_wsj.database import SmsWsj, AudioReader, scenario_map_fn",
        ],
        "dataset": [
            returnn_datasets.SequenceBuffer,
            returnn_datasets.ZipAudioReader,
            returnn_datasets.SmsWsjBase,
            returnn_datasets.SmsWsjBaseWithHdfClasses,
            returnn_datasets.SmsWsjWrapper,
            returnn_datasets.SmsWsjMixtureEarlyDataset,
            returnn_datasets.SmsWsjMixtureEarlyAlignmentDataset,
            segment_to_rasr_librispeech_original,
        ],
    }

    _, dev_data_inputs, test_data_inputs = get_data_inputs(
        use_augmented_lexicon=False, add_unknown_phoneme_and_mapping=False
    )
    scoring_corpora = get_scoring_corpora()

    return SMSHybridSetupData(
        train_key="sms-train-other-960",
        dev_keys=list(dev_data_inputs.keys()),
        test_keys=list(test_data_inputs.keys()),
        align_keys=[],
        train_data_config=datasets["train"],
        cv_data_config=datasets["cv"],
        data_inputs={**dev_data_inputs, **test_data_inputs},
        scoring_corpora=scoring_corpora,
        python_prolog=python_prolog,
        num_classes=num_classes,
    )
