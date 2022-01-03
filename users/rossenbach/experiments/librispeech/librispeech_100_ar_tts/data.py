from sisyphus import tk

from i6_core.corpus import CorpusToTxtJob, CorpusReplaceOrthFromTxtJob
from i6_core.lib import lexicon
from i6_core.text import PipelineJob
from i6_core.lexicon.modification import WriteLexiconJob, MergeLexiconJob


from i6_experiments.common.datasets.librispeech import get_g2p_augmented_bliss_lexicon_dict, get_corpus_object_dict
from i6_experiments.users.rossenbach.datasets.common import DatasetGroup
from i6_experiments.users.rossenbach.datasets.librispeech import get_librispeech_tts_segments
from i6_experiments.users.rossenbach.audio.silence_removal import AlignmentCacheSilenceRemoval
from i6_experiments.users.rossenbach.corpus.transform import ApplyLexiconToTranscriptions

from ..librispeech_100_gmm.unfolded_monophon_baseline import get_monophone_ls100_training_alignment_and_allophones,\
    align_any_data

FFMPEG_BINARY = tk.Path("/u/rossenbach/bin/ffmpeg", hash_overwrite="FFMPEG_BINARY")


def get_static_lexicon():
    """
    Add the phoneme and lemma entries for special TTS symbols

    :param bool include_punctuation:
    :return: the lexicon with special lemmas and phonemes
    :rtype: lexicon.Lexicon
    """
    lex = lexicon.Lexicon()

    lex.add_lemma(
        lexicon.Lemma(orth=["[space]"], phon=["[space]"])
    )
    lex.add_phoneme("[space]", variation="none")

    lex.add_lemma(
        lexicon.Lemma(orth=["[start]"], phon=["[start]"])
    )
    lex.add_phoneme("[start]", variation="none")

    lex.add_lemma(
        lexicon.Lemma(orth=["[end]"], phon=["[end]"])
    )
    lex.add_phoneme("[end]", variation="none")

    return lex


def get_lexicon():
    """
    creates the TTS specific Librispeech-100 lexicon

    :return:
    """
    librispeech_100_lexicon = get_g2p_augmented_bliss_lexicon_dict()["train-clean-100"]
    static_bliss_lexcion = WriteLexiconJob(get_static_lexicon()).out_bliss_lexicon
    lexicon = MergeLexiconJob(bliss_lexica=[static_bliss_lexcion, librispeech_100_lexicon],
                              sort_phonemes=True,
                              sort_lemmata=False).out_bliss_lexicon
    return lexicon


def get_silence_processed_dataset_group():
    """
    :return: a dataset group with the ls100 corpus divided into tts-train and tts-dev
    (overlapping speakers, 4 segments per speaker)
    :rtype: DatasetGroup
    """
    corpus_object_dict = get_corpus_object_dict(audio_format="wav")

    train_clean_100 = corpus_object_dict['train-clean-100']
    _, allophones = get_monophone_ls100_training_alignment_and_allophones()
    alignment  = align_any_data(train_clean_100, "train-clean-100-tts", 10, lexicon=None, uncached=False)
    silence_removal_job = AlignmentCacheSilenceRemoval(
        bliss_corpus=train_clean_100.corpus_file,
        alignment_cache=alignment,
        allophone_file=allophones,
        window_shift=0.01,
        pause_duration=0.5,
        output_format="ogg",
        ffmpeg_binary=FFMPEG_BINARY
    )
    processed_train_clean_100 = silence_removal_job.out_corpus
    train_clean_100.corpus_file = processed_train_clean_100
    train_clean_100.audio_format = "ogg"

    dataset_group = DatasetGroup("librispeech-100-tts.silenceprocessed.ogg")
    dataset_group.add_corpus_object("train-clean-100", train_clean_100)

    train_segments, dev_segments = get_librispeech_tts_segments()

    dataset_group.add_segmented_dataset("train-clean-100-tts-train", "train-clean-100", train_segments)
    dataset_group.add_segmented_dataset("train-clean-100-tts-dev", "train-clean-100", dev_segments)

    dataset_group = dataset_group.apply_bliss_processing_function(process_corpus_text_with_lexicon,
                                                                  args={'lexicon': get_lexicon()},
                                                                  new_name="librispeech-100-tts.silenceprocessed.ogg.phon")

    return dataset_group


def process_corpus_text_with_lexicon(bliss_corpus, lexicon):
    """
    :param tk.Path bliss_corpus:
    :param tk.Path lexicon:
    :return: bliss_corpus
    :rtype: tk.Path
    """
    bliss_text = CorpusToTxtJob(bliss_corpus).out_txt

    add_start_command = "sed 's/^/[start] /g'"
    add_end_command = "sed 's/$/ [end]/g'"

    tokenized_text = PipelineJob(
        bliss_text,
        [add_start_command,
         add_end_command]).out
    processed_bliss_corpus = CorpusReplaceOrthFromTxtJob(bliss_corpus, tokenized_text).out_corpus

    converted_bliss_corpus = ApplyLexiconToTranscriptions(processed_bliss_corpus, lexicon, word_separation_orth="[space]").out_corpus
    return converted_bliss_corpus
