from sisyphus import tk
import copy
from typing import Dict, List, Tuple, Optional, Union
from i6_core.corpus import FilterCorpusBySegmentsJob, FilterSegmentsByListJob, SegmentCorpusJob
import i6_experiments.common.datasets.switchboard as swb_dataset
from i6_experiments.common.datasets.switchboard.constants import concurrent
from i6_experiments.common.setups.rasr import util as rasr_util

from i6_experiments.users.berger.recipe.corpus.transform import TransformTranscriptionsJob, TranscriptionTransform
from i6_experiments.users.berger.corpus.general.helpers import filter_unk_in_corpus_object
from i6_experiments.users.berger import helpers
from i6_experiments.users.berger.recipe.lexicon.modification import (
    EnsureSilenceFirstJob,
    DeleteEmptyOrthJob,
    MakeBlankLexiconJob,
)

from i6_experiments.users.raissi.experiments.swb.legacy.data_preparation.lm_data import get_lm


def get_data_inputs(
    train_key: str = "train",
    cv_keys: Optional[List[str]] = None,
    dev_keys: Optional[List[str]] = None,
    test_keys: Optional[List[str]] = None,
    lm_names: Optional[List[str]] = None,
    filter_unk_from_corpus: bool = True,
    filter_short_segments: bool = True,
    add_all_allophones: bool = False,
) -> Tuple[Dict[str, helpers.RasrDataInput], ...]:
    if cv_keys is None:
        cv_keys = ["hub5e00"]
    if dev_keys is None:
        dev_keys = ["hub5e00"]
    if test_keys is None:
        test_keys = ["hub5e01"]
    if lm_names is None:
        lm_names = ["zoltan_4gram"]

    corpus_object_dict = {
        "train": swb_dataset.get_train_corpus_object_i6_legacy(),
        "hub5e00": swb_dataset.get_hub5e00_corpus_object(),
        "hub5e01": swb_dataset.get_hub5e01_corpus_object(),
        "rt03s": swb_dataset.get_rt03s_corpus_object(),
    }

    lms = {lm_name: get_lm(lm_name) for lm_name in lm_names}
    bliss_lexicon = swb_dataset.get_bliss_lexicon()
    bliss_lexicon = EnsureSilenceFirstJob(bliss_lexicon).out_lexicon

    lexicon_config = helpers.LexiconConfig(
        filename=bliss_lexicon,
        normalize_pronunciation=False,
        add_all_allophones=add_all_allophones,
        add_allophones_from_lexicon=not add_all_allophones,
    )

    train_data_inputs = {}
    cv_data_inputs = {}
    dev_data_inputs = {}
    test_data_inputs = {}

    train_corpus_object = copy.deepcopy(corpus_object_dict[train_key])
    if filter_unk_from_corpus:
        filter_unk_in_corpus_object(train_corpus_object, bliss_lexicon)

    final_segment_files = SegmentCorpusJob(train_corpus_object.corpus_file, 1).out_single_segment_files
    if filter_short_segments:
        filtered_segment_files = FilterSegmentsByListJob( #too short not good for 40ms training
            final_segment_files,
            filter_list=[
                "switchboard-1/sw04118A/sw4118A-ms98-a-0045",
                "switchboard-1/sw02663A/sw2663A-ms98-a-0022",
                "switchboard-1/sw02986A/sw2986A-ms98-a-0013",
            ],
        ).out_single_segment_files
        final_segment_files = filtered_segment_files[1]
    train_corpus_object.corpus_file = FilterCorpusBySegmentsJob(
        train_corpus_object.corpus_file, segment_file=final_segment_files
    ).out_corpus

    train_data_inputs[train_key] = helpers.RasrDataInput(
        corpus_object=train_corpus_object,
        concurrent=concurrent[train_key],
        lexicon=lexicon_config,
    )

    for cv_key in cv_keys:
        cv_corpus_object = copy.deepcopy(corpus_object_dict[cv_key])
        if filter_unk_from_corpus:
            filter_unk_in_corpus_object(cv_corpus_object, bliss_lexicon)

        final_segment_files = SegmentCorpusJob(cv_corpus_object.corpus_file, 1).out_single_segment_files
        if filter_short_segments:
            filtered_segment_files = FilterSegmentsByListJob( #too short not good for 40ms training
                final_segment_files,
                filter_list=[
                    "hub5e_00/en_6189a/36",
                    "hub5e_00/en_4852b/77",
                    "hub5e_00/en_6189b/66",
                ],
            ).out_single_segment_files
            final_segment_files = filtered_segment_files[1]
        cv_corpus_object.corpus_file = FilterCorpusBySegmentsJob(
            cv_corpus_object.corpus_file, segment_file=final_segment_files
        ).out_corpus

        cv_corpus_object.corpus_file = TransformTranscriptionsJob(
            cv_corpus_object.corpus_file, TranscriptionTransform.ALL_LOWERCASE
        ).out_corpus_file
        cv_data_inputs[cv_key] = helpers.RasrDataInput(
            corpus_object=cv_corpus_object,
            concurrent=concurrent[cv_key],
            lexicon=lexicon_config,
        )

    for dev_key in dev_keys:
        for lm_name, lm in lms.items():
            dev_data_inputs[f"{dev_key}_{lm_name}"] = helpers.RasrDataInput(
                corpus_object=corpus_object_dict[dev_key],
                concurrent=concurrent[dev_key],
                lexicon=lexicon_config,
                lm=lm,
            )

    for test_key in test_keys:
        for lm_name, lm in lms.items():
            test_data_inputs[f"{test_key}_{lm_name}"] = helpers.RasrDataInput(
                corpus_object=corpus_object_dict[test_key],
                concurrent=concurrent[test_key],
                lexicon=lexicon_config,
                lm=lm,
            )

    return train_data_inputs, cv_data_inputs, dev_data_inputs, test_data_inputs


def get_init_args(
    *,
    dc_detection: bool = False,
    am_extra_args: Optional[Dict] = None,
    mfcc_filter_width: Optional[Union[float, Dict]] = None,
    mfcc_cepstrum_options: Optional[Dict] = None,
    mfcc_extra_args: Optional[Dict] = None,
    gt_normalization: bool = True,
    gt_options_extra_args: Optional[Dict] = None,
):
    """
    :param dc_detection:
    :param am_extra_args:
    :param mfcc_filter_width: dict(channels=20, warping_function="mel", f_max=8000, f_min=0) or 268.258
    :param mfcc_cepstrum_options:
    :param mfcc_extra_args:
    :param gt_normalization:
    :param gt_options_extra_args:
    :return:
    """

    costa_args = {"eval_recordings": False, "eval_lm": False}

    feature_extraction_args = {
        "gt": None

    }

    am_args = {
        "state_tying": "monophone",
        "states_per_phone": 3,
        "state_repetitions": 1,
        "across_word_model": True,
        "early_recombination": False,
        "tdp_scale": 1.0,
        "tdp_transition": (3.0, 0.0, "infinity", 0.0),  # loop, forward, skip, exit
        "tdp_silence": (0.0, 3.0, "infinity", 20.0),
        "tying_type": "global",
        "nonword_phones": "",
        "tdp_nonword": (
            0.0,
            3.0,
            "infinity",
            21.0,
        ),  # only used when tying_type = global-and-nonword
    }
    if am_extra_args is not None:
        am_args.update(am_extra_args)




    #ToDo check feature extraction
    return rasr_util.RasrInitArgs(
        costa_args=costa_args,
        am_args=am_args,
        feature_extraction_args=feature_extraction_args,
    )

