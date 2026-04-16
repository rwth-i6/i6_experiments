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
from i6_core.audio import BlissChangeEncodingJob
from i6_core.corpus import FilterCorpusBySegmentsJob

from i6_experiments.common.datasets.util import CorpusObject
from i6_experiments.common.baselines.librispeech.data import CorpusData
import i6_experiments.common.setups.rasr.util as rasr_util
from i6_experiments.common.setups.rasr import (
    RasrDataInput,
    RasrInitArgs,
)
from i6_experiments.users.raissi.setups.common.data.pipeline_helpers import InputKey
from i6_experiments.users.raissi.args.rasr.features.init_args import get_feature_extraction_args_16kHz
from i6_experiments.users.raissi.args.rasr.am.init_args import get_init_am_args
from i6_experiments.users.raissi.setups.loquacious.config import (
    CV_SEGMENTS
)


from IPython import embed


class DatasetSize(Enum):
    S = "train.small"
    M = "train.medium"
    L = "train.large"

    def __str__(self):
        return self.value

SIS_PREPATH = "/work/common/asr/loquacious/sisyphus_export_setup"
DATA_PREPATH = "/work/asr3/raissi/data/loquacious"

#### keys ####
DEV_KEYS = ["dev.commonvoice", "dev.voxpopuli", "dev.yodas", "dev.librispeech", "dev.short"]
TEST_KEYS = ["test.commonvoice", "test.voxpopuli", "test.yodas", "test.librispeech"]



### Lexica ###
def get_lexica_path(dataset_size: DatasetSize = DatasetSize.S, version=1, single_pron=False):
    return {
        "train": tk.Path(
            ("/").join([DATA_PREPATH, f"lexica/{dataset_size.value}/v{version}/train{'.singlepron' if single_pron else ''}.oov.lexicon.gz"]),
            cached=True,
            hash_overwrite=f"trainlex{'.singlepron' if single_pron else ''}{version}{dataset_size.value}",
        ),
        "recog": tk.Path(
            ("/").join([DATA_PREPATH, f"lexica/{dataset_size.value}/v{version}/lexicon.xml.gz"]),
            cached=True,
            hash_overwrite=f"recoglex{version}{dataset_size.value}",
        ),
    }

### Corpora ###
def get_sis_corpus_path(name):
    return tk.Path(
        ("/").join([SIS_PREPATH, f"output/loquacious.{name}.xml.gz"]), cached=True, hash_overwrite=f"corpus{name}v1"
    )

### LM ###
def get_lm_path(version=1):
    return tk.Path(
        ("/").join([DATA_PREPATH, f"lm/v{version}/4gram-pruned.arpa.gz"]), cached=True, hash_overwrite=f"lm{version}"
    )
#################
DURATIONS = {
    "train.small": 250.0,
    "train.medium": 2500.0,
    "dev.all": 16.5,
    "dev.commonvoice": 5.0,
    "dev.librispeech": 5.0,
    "dev.voxpopuli": 5.0,
    "dev.yodas": 1.5,
    "test.all": 16.5,
    "test.commonvoice": 5.0,
    "test.librispeech": 5.0,
    "test.voxpopuli": 5.0,
    "test.yodas": 1.5,
    # for tuning purposes
    "dev.short": 6.0,
}

CONCURRENT = {
    "train.small": 50,
    "train.medium": 100,
}


# -------------------- functions --------------------
def _get_corpus_object_dict(corpus_name, audio_output_format="wav"):
    audio_dir = None
    if corpus_name != "dev.short":
        bliss_corpus = get_sis_corpus_path(corpus_name)
    else:
        bliss_corpus = FilterCorpusBySegmentsJob(
            bliss_corpus=get_sis_corpus_path("dev.all"), segment_file=CV_SEGMENTS, delete_empty_recordings=True
        ).out_corpus

    if audio_output_format != "ogg":
        conversion_job = BlissChangeEncodingJob(
            corpus_file=bliss_corpus, output_format=audio_output_format, sample_rate=16000
        )
        if corpus_name == DatasetSize.M.value:
            conversion_job.rqmt["time"] = 100
        bliss_corpus = conversion_job.out_corpus
        audio_dir = conversion_job.out_audio_folder

    corpus_object = CorpusObject()
    corpus_object.corpus_file = bliss_corpus
    corpus_object.audio_format = audio_output_format
    corpus_object.audio_dir = audio_dir
    corpus_object.duration = DURATIONS[corpus_name]

    return corpus_object


def get_corpus_data_inputs(
    dataset_size: DatasetSize = DatasetSize.S,
    lm_version: int = 1,
    lexicon_version: int = 1,
    audio_output_format="wav"
) -> CorpusData:

    lm_data = {
        "filename": get_lm_path(version=lm_version),
        "type": "ARPA",
        "scale": 10,
    }

    lexica = get_lexica_path(dataset_size=dataset_size, version=lexicon_version)
    train_lexicon = {
        "filename": lexica["train"],
        "normalize_pronunciation": False,
    }
    recog_lexicon = {
        "filename": lexica["recog"],
        "normalize_pronunciation": False,
    }

    ### create CorpusData ###
    train_data_inputs = {}
    dev_data_inputs = {}
    test_data_inputs = {}

    train_data_inputs[dataset_size.value] = RasrDataInput(
        corpus_object=_get_corpus_object_dict(dataset_size.value, audio_output_format=audio_output_format),
        concurrent=CONCURRENT[dataset_size.value],
        lexicon=train_lexicon,
        lm=None,
    )
    for dev_key in DEV_KEYS:
        dev_data_inputs[dev_key] = RasrDataInput(
            corpus_object=_get_corpus_object_dict(dev_key, audio_output_format=audio_output_format),
            concurrent=12,
            lexicon=recog_lexicon,
            lm=lm_data,
        )

    for test_key in TEST_KEYS:
        test_data_inputs[test_key] = RasrDataInput(
            corpus_object=_get_corpus_object_dict(test_key, audio_output_format=audio_output_format),
            concurrent=12,
            lexicon=recog_lexicon,
            lm=lm_data,
        )

    return CorpusData(
        train_data=train_data_inputs,
        dev_data=dev_data_inputs,
        test_data=test_data_inputs,
    )


def get_init_args(
    *,
    dc_detection: bool = False,
    am_extra_args: Optional[Dict] = None,
    mfcc_cepstrum_options: Optional[Dict] = None,
    mfcc_extra_args: Optional[Dict] = None,
    gt_options_extra_args: Optional[Dict] = None,
):
    """
    :param dc_detection:
    :param am_extra_args:
    :param mfcc_cepstrum_options:
    :param mfcc_extra_args:
    :param gt_options_extra_args:
    :return:
    """

    am_args = get_init_am_args()
    if am_extra_args is not None:
        am_args.update(am_extra_args)

    costa_args = {"eval_recordings": True, "eval_lm": False}

    feature_extraction_args = get_feature_extraction_args_16kHz(
        dc_detection=dc_detection,
        mfcc_cepstrum_options=mfcc_cepstrum_options,
        mfcc_args=mfcc_extra_args,
        gt_args=gt_options_extra_args,
    )
    return RasrInitArgs(
        costa_args=costa_args,
        am_args=am_args,
        feature_extraction_args=feature_extraction_args,
    )

