import copy
import itertools
from typing import Any, List, Optional, Tuple, Dict
from sisyphus import tk

from i6_core.corpus import SegmentCorpusJob
from i6_core.corpus import FilterCorpusRemoveUnknownWordSegmentsJob

from i6_experiments.common.setups.rasr.util import RasrDataInput, ReturnnRasrDataInput
from i6_experiments.users.berger.args.jobs.rasr_init_args import get_am_config_args


def get_returnn_rasr_data_input(
    rasr_data: RasrDataInput,
    name: str = "init",
    segment_path: Optional[str] = None,
    am_args: Optional[Dict] = None,
    allophone_file: Optional[tk.Path] = None,
    concurrent: Optional[int] = None,
    **kwargs,
) -> ReturnnRasrDataInput:

    data_input = ReturnnRasrDataInput(name=name, **kwargs)

    if concurrent is None:
        concurrent = rasr_data.concurrent

    if segment_path is None:
        segment_path = SegmentCorpusJob(
            rasr_data.corpus_object.corpus_file, concurrent
        ).out_segment_path

    if am_args is None:
        am_args = {}

    data_input.get_crp(
        am_args=get_am_config_args(am_args),
        corpus_object=rasr_data.corpus_object,
        concurrent=concurrent,
        segment_path=segment_path,
        lexicon_args=rasr_data.lexicon,
        lm_args=rasr_data.lm,
        allophone_file=allophone_file,
    )

    return data_input


def get_returnn_rasr_data_inputs(
    train_data_inputs: Dict[str, RasrDataInput],
    cv_data_inputs: Dict[str, RasrDataInput],
    dev_data_inputs: Dict[str, RasrDataInput] = {},
    test_data_inputs: Dict[str, RasrDataInput] = {},
    align_data_inputs: Dict[str, RasrDataInput] = {},
    feature_flows: Optional[Dict[str, Any]] = None,
    feature_caches: Optional[Dict[str, Any]] = None,
    f_name: str = "gt",
    train_cv_pairing: Optional[List[Tuple[str, str]]] = None,
    am_args: Optional[dict] = None,
    alignments: Optional[Dict[str, tk.Path]] = None,
    allophone_file: Optional[tk.Path] = None,
) -> Dict[str, dict]:

    train_cv_pairing = train_cv_pairing or list(
        itertools.product(list(train_data_inputs.keys()), list(cv_data_inputs.keys()))
    )
    alignments = alignments or {}
    feature_flows = feature_flows or {}
    feature_caches = feature_caches or {}

    nn_data_inputs = {"train": {}, "cv": {}, "dev": {}, "test": {}, "align": {}}

    for train_key, cv_key in train_cv_pairing:
        nn_data_inputs["train"][f"{train_key}.train"] = get_returnn_rasr_data_input(
            train_data_inputs[train_key],
            feature_flow=feature_flows.get(train_key, {}).get(f_name, None),
            features=feature_caches.get(train_key, {}).get(f_name, None),
            alignments=alignments.get(f"{train_key}_align", None),
            am_args=am_args,
            shuffle_data=True,
            allophone_file=allophone_file,
            # concurrent=1,
        )

        # Set CV lexicon to train lexicon
        cv_data_input = copy.deepcopy(cv_data_inputs[cv_key])
        cv_data_input.lexicon = train_data_inputs[train_key].lexicon

        nn_data_inputs["cv"][f"{train_key}.cv"] = get_returnn_rasr_data_input(
            cv_data_input,
            feature_flow=feature_flows.get(cv_key, {}).get(f_name, None),
            features=feature_caches.get(cv_key, {}).get(f_name, None),
            alignments=alignments.get(f"{cv_key}_align", None),
            am_args=am_args,
            shuffle_data=False,
            allophone_file=allophone_file,
            # concurrent=1,
        )

        # Remove segments with unknown words from cv corpus
        nn_data_inputs["cv"][
            f"{train_key}.cv"
        ].crp.corpus_config.file = FilterCorpusRemoveUnknownWordSegmentsJob(
            nn_data_inputs["cv"][f"{train_key}.cv"].crp.corpus_config.file,
            nn_data_inputs["cv"][f"{train_key}.cv"].crp.lexicon_config.file,
            case_sensitive=True,
        ).out_corpus

        for dev_key in dev_data_inputs:
            nn_data_inputs["dev"][dev_key] = get_returnn_rasr_data_input(
                dev_data_inputs[dev_key],
                feature_flow=feature_flows.get(dev_key, {}).get(f_name, None),
                features=feature_caches.get(dev_key, {}).get(f_name, None),
                am_args=am_args,
                allophone_file=allophone_file,
                shuffle_data=False,
            )

        for test_key in test_data_inputs:
            nn_data_inputs["test"][test_key] = get_returnn_rasr_data_input(
                test_data_inputs[test_key],
                feature_flow=feature_flows.get(test_key, {}).get(f_name, None),
                features=feature_caches.get(test_key, {}).get(f_name, None),
                am_args=am_args,
                allophone_file=allophone_file,
                shuffle_data=False,
            )

        for align_key in align_data_inputs:
            nn_data_inputs["align"][align_key] = get_returnn_rasr_data_input(
                align_data_inputs[align_key],
                feature_flow=feature_flows.get(align_key, {}).get(f_name, None),
                features=feature_caches.get(align_key, {}).get(f_name, None),
                am_args=am_args,
                allophone_file=allophone_file,
                shuffle_data=False,
            )

    return nn_data_inputs
