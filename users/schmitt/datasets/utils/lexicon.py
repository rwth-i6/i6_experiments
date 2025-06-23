from typing import Optional, Sequence

from sisyphus import Job, tk, Task

from i6_core.lib import lexicon

from i6_experiments.common.datasets.librispeech.lexicon import get_g2p_augmented_bliss_lexicon_dict

from returnn_common.datasets_old_2022_10.interface import VocabConfigStatic


def get_phon_vocab(
  num_classes: int, unknown_label: Optional[str] = None, extra_labels: Sequence[str] = (), **other
):
  """
  Get phoneme vocabulary by extracting it from a LibriSpeech lexicon.

  :param num_classes: number of classes. Unfortunately you must know this in advance currently.
      So you could run the pipeline with some dummy value, and then it will crash,
      but then you will see the correct value,
      and fix this.
      Later, we would make this a Sisyphus Variable, but currently our pipeline does not really allow this.
  :param unknown_label: None (default) means there is no unknown label
  :param extra_labels: additional labels to add to the beginning, e.g. BOS/EOS.
      (currently I tend to use "\n" as EOS).
  :param other: passed to :class:`VocabConfigStatic` opts
  """

  phon_lexicon = get_g2p_augmented_bliss_lexicon_dict(add_unknown_phoneme_and_mapping=False)
  phon_vocab = VocabFromLexiconJob(phon_lexicon["train-other-960"]).out_vocab

  return VocabConfigStatic(
    num_classes=num_classes,
    opts={"class": "CharacterTargets", "vocab_file": phon_vocab, "unknown_label": unknown_label, **other},
  )


class VocabFromLexiconJob(Job):
  def __init__(self, lexicon_path: tk.Path):
    self.input_lexicon_path = lexicon_path

    self.out_vocab = self.output_path("phon.vocab")

  def tasks(self):
    yield Task("run", mini_task=True)

  def run(self):
    input_lexicon = lexicon.Lexicon()
    input_lexicon.load(self.input_lexicon_path.get_path())

    for phoneme in input_lexicon.phonemes:
      print(f"Phoneme: {phoneme}")
