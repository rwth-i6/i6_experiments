from typing import List, Optional, Union
from sisyphus import tk

from i6_core.corpus import MergeCorporaJob, MergeStrategy
from i6_core.features import filter_width_from_channels

from i6_experiments.users.hilmes.common.setups.rasr.util import ForcedAlignmentArgs
from i6_experiments.common.datasets.tedlium2.lexicon import get_bliss_lexicon
from i6_experiments.common.helpers.g2p import G2PBasedOovAugmenter


def get_align_dev_args(
    crp: Union[tk.Path, List[tk.Path]], name: str = "nn-cv", target_corpus_keys: Optional[List[str]] = None
) -> ForcedAlignmentArgs:

    if target_corpus_keys is None:
        target_corpus_keys = ["nn-cv"]

    alias_path = "g2p_forced_alignment"

    kernel_lexicon = get_bliss_lexicon()

    if isinstance(crp, list):
        crp = MergeCorporaJob(
            bliss_corpora=crp,
            name=name,
            merge_strategy=MergeStrategy.FLAT,
        ).out_merged_corpus

    g2p_augmenter = G2PBasedOovAugmenter(
        original_bliss_lexicon=kernel_lexicon,
        train_lexicon=kernel_lexicon,
    )
    forced_align_lexicon = g2p_augmenter.get_g2p_augmented_bliss_lexicon(
        bliss_corpus=crp,
        corpus_name="dev-clean-other",
        alias_path=alias_path,
    )

    return ForcedAlignmentArgs(
        name=name,
        target_corpus_keys=target_corpus_keys,
        flow="uncached_mfcc+context+lda+vtln+cmllr",  # TODO??
        feature_scorer="train_vtln+sat",
        scorer_index=-1,
        bliss_lexicon={
            "filename": forced_align_lexicon,
            "normalize_pronunciation": False,
        },
        dump_alignment=True,
    )


def get_gammatone_feature_extraction_args():
    return {
        "gt_options": {
            "minfreq": 100,
            "maxfreq": 7500,
            "channels": 50,
            "tempint_type": "hanning",
            "tempint_shift": 0.01,
            "tempint_length": 0.025,
            "flush_before_gap": True,
            "do_specint": False,
            "specint_type": "hanning",
            "specint_shift": 4,
            "specint_length": 9,
            "normalize": True,
            "preemphasis": True,
            "legacy_scaling": False,
            "without_samples": False,
            "samples_options": {
                "audio_format": "wav",
                "dc_detection": False,
            },
            "normalization_options": {},
        }
    }


def get_log_mel_feature_extraction_args():

    return {
        "filterbank_options": {
            "warping_function": "mel",
            "filter_width": filter_width_from_channels(channels=20, warping_function="mel", f_max=8000),
            "normalize": True,
            "normalization_options": None,
            "without_samples": False,
            "samples_options": {
                "audio_format": "wav",
                "dc_detection": False,
            },
            "fft_options": None,
            "add_features_output": True,
            "apply_log": True,
            "add_epsilon": True,
        }
    }
