import copy
from typing import Dict, List, Tuple, Optional, Union
from IPython import embed

from sisyphus import tk
from i6_core.corpus import FilterCorpusBySegmentsJob, FilterSegmentsByListJob, SegmentCorpusJob

from i6_experiments.common.setups.rasr import util as rasr_util
from i6_experiments.common.baselines.librispeech.data import CorpusData

from i6_experiments.users.raissi.setups.common.corpus.corpus_object import get_corpus_object


from i6_experiments.users.raissi.experiments.swb.legacy.data_preparation.lm_data import get_lm
from i6_experiments.users.raissi.experiments.swb.legacy.data_preparation.legacy_constants_and_paths_swb1 import (
    concurrent,
    corpora,
    lexica,
    stm_path,
    glm_path,
)


def get_swb_corpus_object(corpus_name, mapping):
    return get_corpus_object(
        bliss_corpus=corpora[corpus_name][mapping].corpus_file,
        duration=corpora[corpus_name][mapping].duration,
        audio_dir=corpora[corpus_name][mapping].audio_dir,
        audio_format=corpora[corpus_name][mapping].audio_format,
    )


def get_lexicon_config(lexicon_path, normalize_pronunciation=False):
    return {
        "filename": lexicon_path,
        "normalize_pronunciation": normalize_pronunciation,
    }


def get_data_inputs_with_paths(
    train_key: str = "train",
    dev_keys: str = ["dev_zoltan"],
    test_keys: str = ["hub5-01"],
    lm_name: str = "fisher_4gram",
    filter_short_segments: bool = True,
    train_concurrent=200,
    segments_to_filter=None,
):

    train_data_inputs = {}
    dev_data_inputs = {}
    test_data_inputs = {}

    if segments_to_filter is None:
        segments_to_filter = [
            "switchboard-1/sw04118A/sw4118A-ms98-a-0045",
            "switchboard-1/sw02663A/sw2663A-ms98-a-0022",
            "switchboard-1/sw02986A/sw2986A-ms98-a-0013",
        ]

    train_corpus_object = get_swb_corpus_object(corpus_name=train_key, mapping="full")
    final_segment_files = SegmentCorpusJob(train_corpus_object.corpus_file, 1).out_single_segment_files

    if filter_short_segments:
        filtered_segment_files = FilterSegmentsByListJob(  # too short not good for 40ms training
            final_segment_files,
            filter_list=segments_to_filter,
        ).out_single_segment_files
        final_segment_files = filtered_segment_files[1]
    train_corpus_object.corpus_file = FilterCorpusBySegmentsJob(
        train_corpus_object.corpus_file, segment_file=final_segment_files
    ).out_corpus

    train_data_inputs[train_key] = rasr_util.RasrDataInput(
        corpus_object=train_corpus_object,
        concurrent=train_concurrent,  # concurrent[train_key],
        lexicon=get_lexicon_config(lexicon_path=lexica[train_key]),
    )
    eval_lm = {
        "filename": get_lm(lm_name).filename,
        "type": "ARPA",
        "scale": 10,
    }
    for dev_key in dev_keys:
        dev_data_inputs["hub500"] = rasr_util.RasrDataInput(
            corpus_object=get_swb_corpus_object(corpus_name="eval", mapping=dev_key),
            concurrent=concurrent["dev"],
            lexicon=get_lexicon_config(lexicon_path=lexica["eval"]),
            lm=eval_lm,
        )

    for test_key in test_keys:
        test_data_inputs["hub501"] = rasr_util.RasrDataInput(
            corpus_object=get_swb_corpus_object(corpus_name="eval", mapping=test_key),
            concurrent=concurrent["eval"],
            lexicon=get_lexicon_config(lexicon_path=lexica["eval"]),
            lm=eval_lm,
        )

    return CorpusData(
        train_data=train_data_inputs,
        dev_data=dev_data_inputs,
        test_data=test_data_inputs,
    )


def get_final_output(name="final", extract_features=False):
    output_args = rasr_util.OutputArgs(name)

    output_args.define_corpus_type("train", "train")
    output_args.define_corpus_type("hub500", "dev")
    output_args.define_corpus_type("hub501", "test")

    output_args.add_feature_to_extract("gt")

    return output_args
