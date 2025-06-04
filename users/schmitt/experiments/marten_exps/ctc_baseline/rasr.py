import re
import os

import i6_experiments.common.datasets.librispeech as lbs_dataset
from i6_experiments.common.datasets.librispeech.lexicon import _get_raw_bliss_lexicon, _get_special_lemma_lexicon
from i6_experiments.users.mueller.datasets.librispeech import _get_bpe_vocab
from i6_experiments.common.helpers.g2p import G2PBasedOovAugmenter

from i6_core.lexicon.bpe import CreateBPELexiconJob
from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.lexicon.modification import WriteLexiconJob, MergeLexiconJob

from sisyphus import tk


def get_phoneme_lexicon_wo_special_dict(
        alias_path: str,
        use_stress_marker=False,
        output_prefix="datasets",
):
  original_bliss_lexicon = _get_raw_bliss_lexicon(use_stress_marker=use_stress_marker, alias_path=alias_path)
  current_bliss_lexicon = original_bliss_lexicon

  augmented_bliss_lexica = {}
  bliss_corpus_dict = lbs_dataset.get_bliss_corpus_dict(output_prefix=output_prefix)
  for corpus_name, bliss_corpus in sorted(bliss_corpus_dict.items()):
    if "train" in corpus_name:
      if corpus_name in ["train-clean-460", "train-other-960"]:
        augmented_bliss_lexica[corpus_name] = current_bliss_lexicon
      else:
        g2p_augmenter = G2PBasedOovAugmenter(
          original_bliss_lexicon=current_bliss_lexicon,
          train_lexicon=original_bliss_lexicon,
        )
        current_bliss_lexicon = g2p_augmenter.get_g2p_augmented_bliss_lexicon(
          bliss_corpus=bliss_corpus,
          corpus_name=corpus_name,
          alias_path=alias_path,
          casing="upper",
        )
        augmented_bliss_lexica[corpus_name] = current_bliss_lexicon

  return augmented_bliss_lexica


def get_bpe_lexicon(
        vocab_str: str,
        train_small: bool,
        add_unknown_phoneme_and_mapping=True,
        add_silence=True,
        use_stress_marker=False,
        output_prefix="datasets",
):
  assert re.match("^bpe[0-9]+.*$", vocab_str)

  alias_path = os.path.join(
    output_prefix,
    "LibriSpeech",
    "%s_lexicon"
    % (
      "regular" if use_stress_marker else "folded",
    ),
  )

  vocab = _get_bpe_vocab(bpe_size=vocab_str[len("bpe"):], train_small=train_small)

  # same as in _get_bpe_vocab
  subword_nmt_job = CloneGitRepositoryJob(
    url="https://github.com/rwth-i6/subword-nmt",
    commit="5015a45e28a958f800ef1c50e7880c0c9ef414cf",
    checkout_folder_name="subword-nmt",
  )

  bpe_lexicon_wo_special_job = CreateBPELexiconJob(
    base_lexicon_path=get_phoneme_lexicon_wo_special_dict(alias_path=alias_path)["train-other-960"],
    bpe_codes=vocab.codes,
    bpe_vocab=vocab.vocab,
    subword_nmt_repo=subword_nmt_job.out_repository
  )
  tk.register_output(f"lexica/{vocab_str}_wo_special", bpe_lexicon_wo_special_job.out_lexicon)

  static_lexicon = _get_special_lemma_lexicon(
    add_unknown_phoneme_and_mapping=False,  # add_unknown_phoneme_and_mapping,
    add_silence=add_silence,
  )
  static_lexicon_job = WriteLexiconJob(static_lexicon, sort_phonemes=True, sort_lemmata=False)

  merge_lexicon_job = MergeLexiconJob(
    bliss_lexica=[
      static_lexicon_job.out_bliss_lexicon,
      bpe_lexicon_wo_special_job.out_lexicon,
    ],
    sort_phonemes=True,
    sort_lemmata=False,
    compressed=True,
  )
  static_lexicon_job.add_alias(os.path.join(alias_path, "static_lexicon_job"))
  merge_lexicon_job.add_alias(os.path.join(alias_path, "merge_lexicon_job"))
  tk.register_output(f"lexica/{vocab_str}", merge_lexicon_job.out_bliss_lexicon)

  return merge_lexicon_job.out_bliss_lexicon
