from i6_core.datasets.switchboard import *
from i6_core.lexicon.conversion import LexiconFromTextFileJob
from i6_core.lexicon.modification import WriteLexiconJob, MergeLexiconJob
from i6_core.lib import lexicon


def get_speakers_list(subdir_prefix=""):
    """
    Returns speakers list

    :param str subdir_prefix: alias name prefix
    :return: Path to switchboard recording to speakers mapping list
    :rtype: tk.Path
    """
    alias_name = os.path.join(subdir_prefix, "Switchboard")

    speakers_stats = DownloadSwitchboardSpeakersStatsJob()
    speakers_stats.add_alias(os.path.join(alias_name, "download_speakres_stats_job"))

    speakers_list = CreateSwitchboardSpeakersListJob(speakers_stats.out_file)
    speakers_list.add_alias(os.path.join(alias_name, "create_speakers_list_job"))

    return speakers_list.out_speakers_list


def get_train_bliss_corpus(audio_dir, subdir_prefix=""):
    """
    Returns Switchboard training bliss corpus

    :param tk.Path audio_dir: path for audio data
    :param str subdir_prefix: alias name prefix
    :return: Path to switchboard training corpus
    :rtype: tk.Path
    """
    alias_name = os.path.join(subdir_prefix, "Switchboard")

    swb_trans_and_dict = DownloadSwitchboardTranscriptionAndDictJob()
    swb_trans_and_dict.add_alias(
        os.path.join(alias_name, "download_trans_and_dict_job")
    )

    speakers_list = get_speakers_list(subdir_prefix=subdir_prefix)
    corpus = CreateSwitchboardBlissCorpusJob(
        audio_dir=audio_dir,
        trans_dir=swb_trans_and_dict.out_trans_dir,
        speakers_list_file=speakers_list,
    )
    corpus.add_alias(os.path.join(alias_name, "create_train_corpus_job"))

    return corpus.out_corpus


def get_special_lemma_lexicon():
    """
    Generate the special phonemes/lemmas for Switchboard

    :rtype lexicon.Lexicon
    """
    lex = lexicon.Lexicon()

    tags = ["[SILENCE]", "[NOISE]", "[VOCALIZED-NOISE]", "[LAUGHTER]"]
    tag_to_phon = {
        "[SILENCE]": "[SILENCE]",
        "[NOISE]": "[NOISE]",
        "[VOCALIZED-NOISE]": "[VOCALIZEDNOISE]",
        "[LAUGHTER]": "[LAUGHTER]",
    }
    for tag in tags:
        lex.add_phoneme(tag_to_phon[tag], variation="none")

    # add non-special lemmas
    for tag in tags[1:]:  # silence is considered below
        lex.add_lemma(
            lexicon.Lemma(
                orth=[tag],
                phon=[tag_to_phon[tag]],
                synt=[[]],
                eval=[[]],
            )
        )

    # create special lemmas
    lex.add_lemma(
        lexicon.Lemma(
            orth=["[SENTENCE-END]"], synt=[["</s>"]], special="sentence-boundary"
        )
    )

    lex.add_lemma(
        lexicon.Lemma(
            orth=["[sentence-begin]"],
            synt=[["<s>"]],
            eval=[[]],
            special="sentence-begin",
        )
    )

    lex.add_lemma(
        lexicon.Lemma(
            orth=["[sentence-end]"], synt=[["</s>"]], eval=[[]], special="sentence-end"
        )
    )

    lex.add_lemma(
        lexicon.Lemma(
            orth=["[SILENCE]", ""],
            phon=["[SILENCE]"],
            synt=[[]],
            eval=[[]],
            special="silence",
        )
    )

    lex.add_lemma(
        lexicon.Lemma(
            orth=["[UNKNOWN]"], synt=[["<unk>"]], eval=[[]], special="unknown"
        )
    )

    return lex


def get_bliss_lexicon(subdir_prefix=""):
    """
    Creates Switchboard bliss xml lexicon. The special lemmas are added via `get_special_lemma_lexicon` function.
    The original raw dictionary is downloaded from here:
    http://www.openslr.org/resources/5/switchboard_word_alignments.tar.gz

    :param str subdir_prefix: alias prefix name
    :return: Path to switchboard bliss lexicon
    :rtype tk.Path
    """
    alias_name = os.path.join(subdir_prefix, "Switchboard")

    static_lexicon = get_special_lemma_lexicon()
    static_lexicon_job = WriteLexiconJob(
        static_lexicon=static_lexicon, sort_phonemes=True, sort_lemmata=False
    )
    static_lexicon_job.add_alias(
        os.path.join(alias_name, "write_special_lemma_lexicon_job")
    )

    raw_lexicon_dir = DownloadSwitchboardTranscriptionAndDictJob()

    # apply preprocessing on words to be consistent with the training corpus
    mapped_raw_lexicon_file = CreateSwitchboardLexiconTextFileJob(
        raw_lexicon_dir.out_raw_dict
    )
    mapped_raw_lexicon_file.add_alias(
        os.path.join(alias_name, "create_lexicon_text_file_job")
    )

    bliss_lexicon = LexiconFromTextFileJob(
        mapped_raw_lexicon_file.out_dict, compressed=True
    )
    bliss_lexicon.add_alias(os.path.join(alias_name, "create_lexicon_job"))

    merge_lexicon_job = MergeLexiconJob(
        bliss_lexica=[
            static_lexicon_job.out_bliss_lexicon,
            bliss_lexicon.out_bliss_lexicon,
        ],
        sort_phonemes=True,
        sort_lemmata=False,
        compressed=True,
    )
    merge_lexicon_job.add_alias(os.path.join(alias_name, "merge_lexicon_job"))

    return merge_lexicon_job.out_bliss_lexicon


def _export_train_bliss_corpus(subdir_prefix=""):
    """
    Registers output for switchboard training corpus based on i6 internal audio directory

    :param str subdir_prefix: alias prefix name
    """
    train_bliss_corpus = get_train_bliss_corpus(
        tk.Path("/u/corpora/speech/switchboard-1/audio"), subdir_prefix=subdir_prefix
    )
    tk.register_output(
        os.path.join(subdir_prefix, "Switchboard/train.corpus.gz"), train_bliss_corpus
    )


def _export_lexicon(subdir_prefix=""):
    """
    Registers output for switchboard lexicon

    :param str subdir_prefix: alias prefix name
    """
    lex = get_bliss_lexicon(subdir_prefix)
    tk.register_output(os.path.join(subdir_prefix, "Switchboard/lexicon.xml.gz"), lex)


def export_all(subdir_prefix=""):
    """
    :param subdir_prefix: alias prefix name
    """
    _export_train_bliss_coprus(subdir_prefix)
    _export_lexicon(subdir_prefix)
