__all__ = ["lexicon_to_hdf"]

import os
import sys
import h5py
import numpy as np
from typing import Optional
from sisyphus import setup_path, gs, tk

import i6_core.returnn as returnn
from returnn.datasets.util.vocabulary import Vocabulary
from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt

sys.path.append("/u/kaloyan.nikolov/experiments/multilang_0325/config")
from i6_experiments.users.nikolov.experiments.voxpopuli.datasets.voxpopuli.lexicon import get_bliss_lexicon, \
    get_bliss_lang_lexicons
from i6_experiments.users.berger.recipe.corpus.transform import ReplaceUnknownWordsJob
from i6_experiments.users.berger.recipe.returnn.hdf import BlissCorpusToTargetHdfJob
from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.default_tools import MINI_RETURNN_ROOT


def vocab_check(hdf_path):
    hdf = h5py.File(hdf_path)
    target = np.array(hdf['inputs'][:hdf['seqLengths'][0][0]]).reshape(1, -1)[0]
    vocabs = ['all.vocab', 'bpe_256.vocab', 'bpe_4989.vocab', 'bpe_512.vocab', 'bpe_5158.vocab', 'corp_mc_5019.vocab',
              'corp_mc_5021.vocab', 'en_es.vocab', 'mc_4943.vocab']
    for v in vocabs:
        vocab = Vocabulary.create_vocab(vocab_file=f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab/{v}",
                                        unknown_label=None)
        if len(vocab.labels) >= max(target):
            print(len(vocab.labels))
            print(max(target))
            result = [vocab.labels[idx] for idx in target if idx != 0]
            print(f"vocab {v}: {result}")
    print(f"max index found: {max(hdf['inputs'])}")


def lexicon_concat_to_hdf(
        # corpus_path: tk.Path,
        lexicon_path: Optional[tk.Path] = None,
        bpe_size: Optional[int] = None,
        subdir_prefix: Optional[str] = None):
    splits = ['train', 'test', 'dev']
    langs = ["cs", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl"]
    if lexicon_path == None:
        lexicon_path = get_bliss_lang_lexicons(
            subdir_prefix=subdir_prefix,
            raw_lexicon_path="/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab",
            full_text_path="/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab",
            bpe_size=bpe_size)

    for lang in langs:
        for split in splits:
            corpus = tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/corpus/asr_{lang}.{split}.corpus.xml.gz")

            j = BlissCorpusToTargetHdfJob(
                bliss_corpus=ReplaceUnknownWordsJob(corpus, lexicon_file=lexicon_path).out_corpus_file,
                bliss_lexicon=lexicon_path,
                returnn_root=MINI_RETURNN_ROOT,
            )

            tk.register_output(
                f"voxpopuli_asr_lexicon_{bpe_size}/{lang}/{split}.hdf",
                j.out_hdf,
            )


def lexicon_to_hdf_vox(
        # corpus_path: tk.Path,
        lexicon_path: Optional[tk.Path] = None,
        bpe_size: Optional[int] = None,
        subdir_prefix: Optional[str] = "vox_4989"):
    splits = ['train', 'test', 'dev']
    langs = ["cs", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl"]
    if lexicon_path == None:
        lexicon_path, _ = get_bliss_lexicon(
            subdir_prefix=subdir_prefix,
            raw_lexicon_path=tk.Path("/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab/lex.txt"),
            full_text_path=tk.Path("/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab/all.txt"),
            bpe_size=bpe_size)

    for lang in langs:
        for split in splits:
            corpus = tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/corpus/asr_{lang}.{split}.corpus.xml.gz")

            j = BlissCorpusToTargetHdfJob(
                bliss_corpus=ReplaceUnknownWordsJob(corpus, lexicon_file=lexicon_path).out_corpus_file,
                bliss_lexicon=lexicon_path,
                returnn_root=MINI_RETURNN_ROOT,
            )

            tk.register_output(
                f"voxpopuli_asr_lexicon_legacy_{bpe_size}/{lang}/{split}.hdf",
                j.out_hdf,
            )


def lexicon_to_hdf(
        corpus_path: tk.Path = None,
        splits=['train', 'test', 'dev'],
        langs=["cs", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl"],
        lexicon_path: tk.Path = None,
        prefix: Optional[str] = None) -> tk.Path:
    if lexicon_path is None:
        lexicon_path = get_bliss_lang_lexicons(
            subdir_prefix="vox_512",
            raw_lexicon_path="/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab",
            full_text_path="/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab",
            bpe_size={'base': 512},
            add_prefix=False)

    corpus_set = corpus_path is not None
    if len(splits) < 3:
        for split in splits:
            print(lexicon_path)
            j = BlissCorpusToTargetHdfJob(
                bliss_corpus=ReplaceUnknownWordsJob(corpus_path, lexicon_file=lexicon_path).out_corpus_file,
                bliss_lexicon=lexicon_path,
                returnn_root=MINI_RETURNN_ROOT,
            )
            tk.register_output(
                f"{prefix}/{split}.hdf",
                j.out_hdf,
            )
    else:
        for lang in langs:
            for split in splits:
                if not corpus_set:
                    corpus_path = tk.Path(
                        f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/corpus/asr_{lang}.{split}.corpus.xml.gz")

                j = BlissCorpusToTargetHdfJob(
                    bliss_corpus=ReplaceUnknownWordsJob(corpus_path, lexicon_file=lexicon_path).out_corpus_file,
                    bliss_lexicon=lexicon_path,
                    returnn_root=MINI_RETURNN_ROOT,
                )

                tk.register_output(
                    f"{prefix}/{lang}/{split}.hdf",
                    j.out_hdf,
                )
    return j.out_hdf
    # vocab_check(j.out_hdf)


def lexicon_to_hdf_miami(
        corpus_path: str = None,
        splits=['test'],
        sets=["full"],
        lexicon_path: tk.Path = None,
        prefix: Optional[str] = None) -> tk.Path:
    if lexicon_path is None:
        lexicon_path = get_bliss_lang_lexicons(
            subdir_prefix="vox_512",
            raw_lexicon_path="/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab",
            full_text_path="/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab",
            bpe_size={'base': 512},
            add_prefix=False)

    corpus_set = corpus_path is not None
    for dataset in sets:
        for split in splits:
            j = BlissCorpusToTargetHdfJob(
                bliss_corpus=ReplaceUnknownWordsJob(tk.Path(corpus_path + f"miami.{dataset}.corpus.xml.gz"),
                                                    lexicon_file=lexicon_path).out_corpus_file,
                bliss_lexicon=lexicon_path,
                returnn_root=MINI_RETURNN_ROOT,
            )
            tk.register_output(
                f"{prefix}/{dataset}/{split}.hdf",
                j.out_hdf,
            )
    return j.out_hdf
