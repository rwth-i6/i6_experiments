import re
import os
from collections import OrderedDict
from ast import literal_eval

import i6_experiments.common.datasets.librispeech as lbs_dataset
from i6_experiments.common.datasets.librispeech.lexicon import _get_raw_bliss_lexicon, _get_special_lemma_lexicon
from i6_experiments.users.mueller.datasets.librispeech import _get_bpe_vocab
from i6_experiments.common.helpers.g2p import G2PBasedOovAugmenter

from i6_core.lexicon.bpe import CreateBPELexiconJob
from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.lexicon.modification import WriteLexiconJob, MergeLexiconJob
import i6_core.rasr as rasr
from i6_core.am.config import acoustic_model_config
from i6_core.lib import lexicon
from i6_core.util import write_xml

from sisyphus import tk, Job, Task


def get_librasr_fsa_config(lexicon_path: tk.Path, corpus_path: tk.Path):
  crp = rasr.CommonRasrParameters()
  rasr.crp_add_default_output(crp)

  crp.corpus_config = rasr.RasrConfig()  # type: ignore
  crp.corpus_config.file = corpus_path  # type: ignore
  crp.corpus_config.capitalize_transcriptions = False
  crp.corpus_config.progress_indication = "global"
  crp.corpus_config.warn_about_unexpected_elements = False

  crp.lexicon_config = rasr.RasrConfig()  # type: ignore
  crp.lexicon_config.file = lexicon_path  # type: ignore
  crp.lexicon_config.normalize_pronunciation = False  # type: ignore

  crp.acoustic_model_config = acoustic_model_config(
    states_per_phone=1,
    tdp_transition=(0.0, 0.0, "infinity", 0.0),  # type: ignore
    tdp_silence=(0.0, 0.0, "infinity", 0.0),  # type: ignore
  )  # type: ignore
  crp.acoustic_model_config.allophones.add_all = False  # type: ignore
  crp.acoustic_model_config.allophones.add_from_lexicon = True  # type: ignore
  crp.acoustic_model_config.tdp.applicator_type = "corrected"  # type: ignore
  crp.acoustic_model_config.tdp.entry_m1.loop = "infinity"  # type: ignore
  crp.acoustic_model_config.tdp.entry_m2.loop = "infinity"  # type: ignore
  crp.acoustic_model_config.fix_allophone_context_at_word_boundaries = True
  crp.acoustic_model_config.transducer_builder_filter_out_invalid_allophones = True

  # Make config from crp
  mapping = {
    "acoustic_model": "lib-rasr.alignment-fsa-exporter.model-combination.acoustic-model",
    "corpus": "lib-rasr.corpus",
    "lexicon": "lib-rasr.alignment-fsa-exporter.model-combination.lexicon",
  }
  config, post_config = rasr.build_config_from_mapping(
    crp,
    mapping,
    parallelize=False,
  )

  config.lib_rasr.alignment_fsa_exporter.allophone_state_graph_builder.orthographic_parser.allow_for_silence_repetitions = False
  config.lib_rasr.alignment_fsa_exporter.allophone_state_graph_builder.orthographic_parser.normalize_lemma_sequence_scores = False

  config_file = rasr.WriteRasrConfigJob(config, post_config).out_config
  tk.register_output("config/librasr_fsa_config", config_file)


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

  reorder_lexicon_job = ReorderLexiconPhonemesJob(
    bliss_lexicon=merge_lexicon_job.out_bliss_lexicon,
    vocab=vocab.vocab,
  )
  tk.register_output(f"lexica/{vocab_str}_reorderd", reorder_lexicon_job.out_bliss_lexicon)

  return reorder_lexicon_job.out_bliss_lexicon


class ReorderLexiconPhonemesJob(Job):
  def __init__(self, bliss_lexicon, vocab):
    self.bliss_lexicon = bliss_lexicon
    self.vocab = vocab

    self.out_bliss_lexicon = self.output_path("out_lexicon.xml.gz")

  def tasks(self):
    yield Task("run", mini_task=True)

  def run(self):
    with open(self.vocab.get_path(), "r") as f:
      vocab_dict = literal_eval(f.read())

    bliss_lexicon = lexicon.Lexicon()
    bliss_lexicon.load(self.bliss_lexicon.get_path())

    bliss_lexicon.phonemes = OrderedDict()

    # add phonemes in the order of the vocab
    for phoneme in vocab_dict:
      if phoneme == "<s>":
        continue

      bliss_lexicon.phonemes[phoneme] = "context"

    # add special silence phoneme at the end
    bliss_lexicon.phonemes["[SILENCE]"] = "none"

    # map SOS lemma to </s> in order to use the same phoneme for SOS and EOS
    for lemma in bliss_lexicon.lemmata:
      if lemma.special == "sentence-begin":
        lemma.synt = ["</s>"]

    write_xml(self.out_bliss_lexicon.get_path(), bliss_lexicon.to_xml())

