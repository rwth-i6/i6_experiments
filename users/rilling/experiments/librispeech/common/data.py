from typing import Optional
from functools import lru_cache

from sisyphus import tk

from i6_core.returnn import BlissToOggZipJob
from returnn_common.datasets import Dataset, OggZipDataset, MetaDataset, HDFDataset

from i6_experiments.users.rossenbach.datasets.librispeech import get_librispeech_tts_segments

from i6_core.corpus.filter import FilterCorpusBySegmentsJob

from i6_experiments.common.datasets.librispeech import (
    get_g2p_augmented_bliss_lexicon_dict,
    get_bliss_lexicon,
    get_bliss_corpus_dict,
)

from i6_experiments.users.rossenbach.setups.tts.preprocessing import (
    process_corpus_text_with_extended_lexicon,
    extend_lexicon_with_tts_lemmas,
    extend_lexicon_with_blank,
)

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import (
    ReturnnAudioRawOptions,
    AudioRawDatastream,
)

def get_librispeech_lexicon(corpus_key="train-clean-100", with_g2p: bool = True, add_silence=True) -> tk.Path:
    """
    get the TTS-extended g2p bliss lexicon with [start], [end] and [space] marker
    :return:
    """
    if with_g2p:
        return extend_lexicon_with_tts_lemmas(
            get_g2p_augmented_bliss_lexicon_dict(
                use_stress_marker=False,
                add_silence=add_silence,
                output_prefix="datasets_tts",
            )[corpus_key],
        )
    else:
        return extend_lexicon_with_tts_lemmas(
            get_bliss_lexicon(use_stress_marker=False, add_silence=add_silence, output_prefix="datasets_tts")
        )

def get_tts_extended_bliss(ls_corpus_key, remove_unk_seqs=False) -> tk.Path:
    """
    get a modified ls corpus using the TTS processing
    :param ls_corpus_key
    :param remove_unk_seqs: remove all sequences with unknowns, used for dev-clean and dev-other
        in case of using them for cross validation
    :return:
    """
    ls_bliss = get_bliss_corpus_dict(audio_format="ogg")[ls_corpus_key]
    if remove_unk_seqs:
        from i6_core.corpus.filter import FilterCorpusRemoveUnknownWordSegmentsJob

        ls_bliss = FilterCorpusRemoveUnknownWordSegmentsJob(
            bliss_corpus=ls_bliss, bliss_lexicon=get_tts_lexicon(), all_unknown=False
        ).out_corpus
    tts_ls_bliss = process_corpus_text_with_extended_lexicon(
        bliss_corpus=ls_bliss, lexicon=get_librispeech_lexicon(corpus_key="train-clean-100")
    )

    return tts_ls_bliss


def get_tts_lexicon(
    with_blank: bool = False, with_g2p: bool = True, corpus_key: str = "train-clean-100", add_silence=True
) -> tk.Path:
    """
    Get the TTS/CTC lexicon

    :param with_blank: add blank (e.g. for CTC training or extraction)
    :return: path to bliss lexicon file
    """
    lexicon = get_librispeech_lexicon(corpus_key=corpus_key, with_g2p=with_g2p, add_silence=add_silence)
    lexicon = extend_lexicon_with_tts_lemmas(lexicon)
    if with_blank:
        lexicon = extend_lexicon_with_blank(lexicon)
    return lexicon


def get_tts_eval_bliss_and_zip(ls_corpus_key, returnn_exe, returnn_root, silence_preprocessed=True, remove_unk_seqs=False):
    """
    :param ls_corpus_key: e.g. train-clean-100, see LibriSpeech data definition
    :param for_training:
    :param silence_preprocessed:
    :param remove_unk_seqs: remove all sequences with unknowns, used for dev-clean and dev-other
        in case of using them for cross validation
    :return:
    """
    bliss_dataset = get_tts_extended_bliss(ls_corpus_key=ls_corpus_key, remove_unk_seqs=remove_unk_seqs)

    zip_dataset = BlissToOggZipJob(
        bliss_corpus=bliss_dataset,
        no_conversion=True,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
    ).out_ogg_zip

    return bliss_dataset, zip_dataset


@lru_cache()
def get_audio_raw_datastream(peak_normalization=False, preemphasis=0.97):
    audio_datastream = AudioRawDatastream(
        available_for_inference=True,
        options=ReturnnAudioRawOptions(peak_normalization=peak_normalization, preemphasis=preemphasis),
    )
    return audio_datastream

def build_swer_test_dataset(synthetic_bliss, returnn_exe, returnn_root, preemphasis: Optional[float] = None, peak_normalization: bool = False):
    """

    :param synthetic_bliss:
    :param preemphasis:
    :param peak_normalization:
    """
    zip_dataset_job = BlissToOggZipJob(
        bliss_corpus=synthetic_bliss,
        no_conversion=True,  # for Librispeech we are already having ogg
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
    )

    audio_datastream = get_audio_raw_datastream(preemphasis, peak_normalization)

    data_map = {"raw_audio": ("zip_dataset", "data")}

    test_zip_dataset = OggZipDataset(
        files=[zip_dataset_job.out_ogg_zip],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        seq_ordering="sorted_reverse",
    )
    test_dataset = MetaDataset(
        data_map=data_map, datasets={"zip_dataset": test_zip_dataset}, seq_order_control_dataset="zip_dataset"
    )

    return test_dataset
