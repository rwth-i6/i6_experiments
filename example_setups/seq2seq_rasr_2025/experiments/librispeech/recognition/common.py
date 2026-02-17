from dataclasses import dataclass, field, replace
from typing import List, Optional, Union

from i6_experiments.common.setups.serialization import Collection
from i6_core.rasr import RasrConfig
from i6_core.returnn import PtCheckpoint
from sisyphus import tk

from ....data.librispeech import datasets as librispeech_datasets
from ....data.librispeech import lm as librispeech_lm
from ....data.librispeech.bpe import get_bpe_vocab_file
from ....data.librispeech.lexicon import get_bliss_phoneme_lexicon, get_bpe_bliss_lexicon
from ....data.librispeech.recog import LibrispeechTreeTimesyncRecogParams
from ....model_pipelines.common.recog import (
    OfflineRecogParameters,
    RecogResult,
    StreamingRecogParameters,
    recog_rasr_offline,
    recog_rasr_streaming,
)
from ....model_pipelines.common.recog_rasr_config import (
    LexiconfreeLabelsyncRecogParams,
    LexiconfreeTimesyncRecogParams,
    get_lexiconfree_labelsync_recog_config,
    get_lexiconfree_timesync_recog_config,
    get_tree_timesync_recog_config,
)


@dataclass
class BaseRecogVariant:
    descriptor: str
    search_algorithm_params: Union[
        LexiconfreeLabelsyncRecogParams, LexiconfreeTimesyncRecogParams, LibrispeechTreeTimesyncRecogParams
    ]
    search_mode_params: Union[OfflineRecogParameters, StreamingRecogParameters] = field(
        default_factory=OfflineRecogParameters
    )
    compute_search_errors: bool = False


def _run_single_variant(
    checkpoint: Optional[PtCheckpoint],
    encoder_serializers: Collection,
    label_scorer_configs: List[RasrConfig],
    vocab_file: Optional[tk.Path],
    lexicon_file: Optional[tk.Path],
    blank_index: Optional[int],
    sentence_end_index: Optional[int],
    variant: BaseRecogVariant,
    corpora: List[librispeech_datasets.EvalSet],
) -> List[RecogResult]:
    if isinstance(variant.search_algorithm_params, LexiconfreeLabelsyncRecogParams):
        assert vocab_file is not None
        if variant.compute_search_errors:
            raise NotImplementedError
        recog_config = get_lexiconfree_labelsync_recog_config(
            vocab_file=vocab_file,
            label_scorer_configs=label_scorer_configs,
            params=variant.search_algorithm_params,
            sentence_end_index=sentence_end_index,
            logfile_suffix="recog",
        )
        align_config = None
    elif isinstance(variant.search_algorithm_params, LexiconfreeTimesyncRecogParams):
        assert vocab_file is not None
        if variant.compute_search_errors:
            raise NotImplementedError
        recog_config = get_lexiconfree_timesync_recog_config(
            vocab_file=vocab_file,
            label_scorer_configs=label_scorer_configs,
            params=variant.search_algorithm_params,
            sentence_end_index=sentence_end_index,
            blank_index=blank_index,
            logfile_suffix="recog",
        )
        align_config = None
    elif isinstance(variant.search_algorithm_params, LibrispeechTreeTimesyncRecogParams):
        assert lexicon_file is not None
        if variant.search_algorithm_params.word_lm_params is not None:
            lm_config = librispeech_lm.get_word_lm_config(
                lexicon_file=lexicon_file, params=variant.search_algorithm_params.word_lm_params
            )
        else:
            lm_config = None
        recog_config = get_tree_timesync_recog_config(
            lexicon_file=lexicon_file,
            label_scorer_configs=label_scorer_configs,
            params=variant.search_algorithm_params,
            lm_config=lm_config,
            logfile_suffix="recog",
        )
        if variant.compute_search_errors:
            align_params = replace(
                variant.search_algorithm_params,
                max_beam_sizes=[2048] * len(variant.search_algorithm_params.max_beam_sizes),
                score_thresholds=[22.0] * len(variant.search_algorithm_params.max_beam_sizes),
                max_word_end_beam_size=None,
                word_end_score_threshold=None,
                maximum_stable_delay=None,
                maximum_stable_delay_pruning_interval=None,
            )
            align_config = get_tree_timesync_recog_config(
                lexicon_file=lexicon_file,
                label_scorer_configs=label_scorer_configs,
                params=align_params,
                lm_config=lm_config,
                logfile_suffix="align",
            )
        else:
            align_config = None

    results = []
    for corpus in corpora:
        recog_data = librispeech_datasets.get_default_recog_data(corpus)
        score_corpus = librispeech_datasets.get_default_score_corpus(corpus)

        if isinstance(variant.search_mode_params, OfflineRecogParameters):
            recog_result = recog_rasr_offline(
                descriptor=variant.descriptor,
                checkpoint=checkpoint,
                recog_rasr_config_file=recog_config,
                align_rasr_config_file=align_config,
                recog_data_config=recog_data,
                recog_corpus=score_corpus,
                encoder_serializers=encoder_serializers,
                sample_rate=16000,
                params=variant.search_mode_params,
            )
        elif isinstance(variant.search_mode_params, StreamingRecogParameters):
            recog_result = recog_rasr_streaming(
                descriptor=variant.descriptor,
                checkpoint=checkpoint,
                recog_rasr_config_file=recog_config,
                recog_data_config=recog_data,
                recog_corpus=score_corpus,
                encoder_serializers=encoder_serializers,
                sample_rate=16000,
                params=variant.search_mode_params,
            )

        results.append(recog_result)

    return results


def run_single_bpe_variant(
    checkpoint: Optional[PtCheckpoint],
    encoder_serializers: Collection,
    label_scorer_configs: List[RasrConfig],
    bpe_size: int,
    blank_index: Optional[int],
    sentence_end_index: Optional[int],
    variant: BaseRecogVariant,
    corpora: List[librispeech_datasets.EvalSet],
) -> List[RecogResult]:
    use_blank = blank_index is not None
    use_sentence_end = sentence_end_index is not None
    vocab_file = get_bpe_vocab_file(bpe_size=bpe_size, add_blank=use_blank)
    lexicon_file = get_bpe_bliss_lexicon(bpe_size=bpe_size, add_blank=use_blank, add_sentence_end_pron=use_sentence_end)

    return _run_single_variant(
        checkpoint=checkpoint,
        encoder_serializers=encoder_serializers,
        label_scorer_configs=label_scorer_configs,
        vocab_file=vocab_file,
        lexicon_file=lexicon_file,
        blank_index=blank_index,
        sentence_end_index=sentence_end_index,
        variant=variant,
        corpora=corpora,
    )


def run_single_phoneme_variant(
    checkpoint: Optional[PtCheckpoint],
    encoder_serializers: Collection,
    label_scorer_configs: List[RasrConfig],
    blank_index: Optional[int],
    sentence_end_index: Optional[int],
    variant: BaseRecogVariant,
    corpora: List[librispeech_datasets.EvalSet],
) -> List[RecogResult]:
    assert not isinstance(variant.search_algorithm_params, LexiconfreeTimesyncRecogParams)
    assert not isinstance(variant.search_algorithm_params, LexiconfreeLabelsyncRecogParams)
    lexicon_file = get_bliss_phoneme_lexicon()

    return _run_single_variant(
        checkpoint=checkpoint,
        encoder_serializers=encoder_serializers,
        label_scorer_configs=label_scorer_configs,
        vocab_file=None,
        lexicon_file=lexicon_file,
        blank_index=blank_index,
        sentence_end_index=sentence_end_index,
        variant=variant,
        corpora=corpora,
    )
