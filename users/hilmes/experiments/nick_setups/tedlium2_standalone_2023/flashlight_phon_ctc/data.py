"""


"""
import os
from sisyphus import tk

from i6_core.corpus.transform import ApplyLexiconToCorpusJob
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from i6_core.returnn.vocabulary import ReturnnVocabFromPhonemeInventory

from i6_experiments.common.datasets.tedlium2.corpus import get_bliss_corpus_dict
from i6_experiments.common.datasets.tedlium2.lexicon import get_g2p_augmented_bliss_lexicon, get_bliss_lexicon
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

from ..data import TrainingDatasetSettings, TrainingDatasets, build_training_datasets, get_zip
from ..data import DATA_PREFIX


def get_eow_lexicon(with_g2p=True) -> tk.Path:
    """
    Standard bliss lexicon modified with EOW
    :return:
    """
    if with_g2p:
        lex = get_g2p_augmented_bliss_lexicon(output_prefix="tedliumv2_g2p_datasets")
    else:
        lex = get_bliss_lexicon(output_prefix="tedliumv2_eow_datasets")

    return AddEowPhonemesToLexiconJob(lex).out_lexicon


def get_eow_text_lexicon() -> tk.Path:
    """

    :return:
    """
    bliss_lex = get_eow_lexicon(with_g2p=False)
    from i6_experiments.users.rossenbach.lexicon.conversion import BlissLexiconToWordLexicon

    word_lexicon = BlissLexiconToWordLexicon(bliss_lex).out_lexicon
    return word_lexicon


def get_eow_bliss(corpus_key, remove_unk_seqs=False) -> tk.Path:
    """
    get an EOW modified corpus with optional unknown removed for cross validation

    :param corpus_key: train, dev, test
    :param remove_unk_seqs: remove all sequences with unknowns, used for dev-clean and dev-other
        in case of using them for cross validation
    :return:
    """
    bliss = get_bliss_corpus_dict(audio_format="wav")[corpus_key]
    if remove_unk_seqs:
        from i6_core.corpus.filter import FilterCorpusRemoveUnknownWordSegmentsJob

        bliss = FilterCorpusRemoveUnknownWordSegmentsJob(
            bliss_corpus=bliss,
            bliss_lexicon=get_eow_lexicon(),  # assume no g2p when removing unknown for test sets
            all_unknown=False,
        ).out_corpus

    # default train lexicon
    lexicon = get_eow_lexicon(with_g2p=True)
    converted_bliss_corpus = ApplyLexiconToCorpusJob(bliss, lexicon, word_separation_orth=None).out_corpus

    return converted_bliss_corpus


def get_eow_bliss_and_zip(corpus_key, remove_unk_seqs=False):
    """
    :param corpus_key: e.g. "train", "dev", or "test,
    :param remove_unk_seqs: remove all sequences with unknowns, used for dev-clean and dev-other
        in case of using them for cross validation
    :return: tuple of bliss and zip
    """

    bliss_dataset = get_eow_bliss(corpus_key=corpus_key, remove_unk_seqs=remove_unk_seqs)
    zip_dataset = get_zip(f"{corpus_key}_eow", bliss_dataset=bliss_dataset)

    return bliss_dataset, zip_dataset


def get_eow_vocab_datastream() -> LabelDatastream:
    """
    Phoneme with EOW LabelDatastream for Tedlium-2

    :param with_blank: datastream for CTC training
    """
    lexicon = get_eow_lexicon()
    blacklist = {"[SILENCE]"}
    returnn_vocab_job = ReturnnVocabFromPhonemeInventory(lexicon, blacklist=blacklist)
    returnn_vocab_job.add_alias(os.path.join(DATA_PREFIX, "eow_returnn_vocab_job"))

    vocab_datastream = LabelDatastream(
        available_for_inference=True, vocab=returnn_vocab_job.out_vocab, vocab_size=returnn_vocab_job.out_vocab_size
    )

    return vocab_datastream


def build_phon_training_datasets(
    settings: TrainingDatasetSettings,
) -> TrainingDatasets:
    """
    :param settings: configuration object for the dataset pipeline
    """
    label_datastream = get_eow_vocab_datastream()

    _, train_ogg = get_eow_bliss_and_zip("train")
    _, dev_ogg = get_eow_bliss_and_zip("dev", remove_unk_seqs=True)

    return build_training_datasets(
        settings=settings, train_ogg=train_ogg, dev_ogg=dev_ogg, label_datastream=label_datastream
    )
