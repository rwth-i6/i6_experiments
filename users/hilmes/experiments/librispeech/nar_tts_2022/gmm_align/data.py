"""
Defines the data inputs for any RASR based LibriSpeech task
"""
from dataclasses import dataclass
from typing import Dict
from i6_core.corpus.filter import FilterCorpusBySegmentsJob

from i6_experiments.common.datasets.librispeech import (
  get_g2p_augmented_bliss_lexicon_dict,
  constants,
  get_arpa_lm_dict,
  get_bliss_lexicon,
)
from i6_experiments.common.setups.rasr.util import RasrDataInput

from i6_experiments.users.rossenbach.datasets.librispeech import (
  get_librispeech_tts_segments,
  get_ls_train_clean_100_tts_silencepreprocessed,
  get_corpus_object_dict,
)
from copy import deepcopy


@dataclass()
class CorpusData:
  """
  Helper class to define all RasrDataInputs to be passed to the `System` class
  """

  train_data: Dict[str, RasrDataInput]
  dev_data: Dict[str, RasrDataInput]
  test_data: Dict[str, RasrDataInput]


def get_corpus_data_inputs():
  """
  :return: a 3-sized tuple containing lists of RasrDataInput for train, dev and test
  """

  # this is the FFmpeg silence preprocessed version of LibriSpeech train-clean-100
  sil_pp_train_clean_100 = get_ls_train_clean_100_tts_silencepreprocessed()
  sil_pp_train_clean_100_align = deepcopy(sil_pp_train_clean_100)

  # segments for train-clean-100-tts-train and train-clean-100-tts-dev
  # (1004 segments for dev, 4 segments for each of the 251 speakers)
  train_segments, dev_segments = get_librispeech_tts_segments()

  # remove the dev segments from the GMM training corpus
  sil_pp_train_clean_100_tts_train_xml = FilterCorpusBySegmentsJob(
    sil_pp_train_clean_100.corpus_file,
    train_segments,
    compressed=True,
  ).out_corpus
  sil_pp_train_clean_100.corpus_file = sil_pp_train_clean_100_tts_train_xml

  g2p_lexica = get_g2p_augmented_bliss_lexicon_dict(
    output_prefix="corpora",
    add_unknown_phoneme_and_mapping=False,
    use_stress_marker=False,
  )
  lm = None
  train_lexicon = {
    "filename": g2p_lexica["train-clean-100"],
    "normalize_pronunciation": False,
  }

  train_data_inputs = {}
  dev_data_inputs = {}
  test_data_inputs = {}

  train_data_inputs["train-clean-100"] = RasrDataInput(
    corpus_object=sil_pp_train_clean_100,
    concurrent=constants.concurrent["train-clean-100"],
    lexicon=train_lexicon,
    lm=lm,
  )

  lm = {
    "filename": get_arpa_lm_dict()["4gram"],
    "type": "ARPA",
    "scale": 10,
  }
  lexicon = {
    "filename": get_bliss_lexicon(
      use_stress_marker=False,
      add_unknown_phoneme_and_mapping=False,
    ),
    "normalize_pronunciation": False,
  }

  corpus_object_dict = get_corpus_object_dict(
    audio_format="wav", output_prefix="corpora"
  )

  for dev_key in ["dev-other"]:
    dev_data_inputs[dev_key] = RasrDataInput(
      corpus_object=corpus_object_dict[dev_key],
      concurrent=constants.concurrent[dev_key],
      lexicon=lexicon,
      lm=lm,
    )

  test_data_inputs["tts_align"] = RasrDataInput(
    corpus_object=sil_pp_train_clean_100_align,
    concurrent=constants.concurrent["train-clean-100"],
    lexicon=train_lexicon,
    lm=lm,
  )

  return CorpusData(
    train_data=train_data_inputs,
    dev_data=dev_data_inputs,
    test_data=test_data_inputs,
  )


def get_synth_corpus_data_inputs(synth_corpus):
  """
  TODO
  :return: a 3-sized tuple containing lists of RasrDataInput for train, dev and test
  """

  # segments for train-clean-100-tts-train and train-clean-100-tts-dev
  # (1004 segments for dev, 4 segments for each of the 251 speakers)
  train_segments, dev_segments = get_librispeech_tts_segments()

  # remove the dev segments from the GMM training corpus
  sil_pp_train_clean_100_tts_train_xml = FilterCorpusBySegmentsJob(
    synth_corpus.corpus_file,
    train_segments,
    compressed=True,
  ).out_corpus
  synth_corpus.corpus_file = sil_pp_train_clean_100_tts_train_xml

  g2p_lexica = get_g2p_augmented_bliss_lexicon_dict(
    output_prefix="corpora",
    add_unknown_phoneme_and_mapping=False,
    use_stress_marker=False,
  )
  lm = None
  train_lexicon = {
    "filename": g2p_lexica["train-clean-100"],
    "normalize_pronunciation": False,
  }

  train_data_inputs = {}
  dev_data_inputs = {}
  test_data_inputs = {}

  train_data_inputs["train-clean-100"] = RasrDataInput(
    corpus_object=synth_corpus,
    concurrent=constants.concurrent["train-clean-100"],
    lexicon=train_lexicon,
    lm=lm,
  )

  lm = {
    "filename": get_arpa_lm_dict()["4gram"],
    "type": "ARPA",
    "scale": 10,
  }
  lexicon = {
    "filename": get_bliss_lexicon(
      use_stress_marker=False,
      add_unknown_phoneme_and_mapping=False,
    ),
    "normalize_pronunciation": False,
  }

  corpus_object_dict = get_corpus_object_dict(
    audio_format="wav", output_prefix="corpora"
  )

  for dev_key in ["dev-other"]:
    dev_data_inputs[dev_key] = RasrDataInput(
      corpus_object=corpus_object_dict[dev_key],
      concurrent=constants.concurrent[dev_key],
      lexicon=lexicon,
      lm=lm,
    )

  return CorpusData(
    train_data=train_data_inputs,
    dev_data=dev_data_inputs,
    test_data=test_data_inputs,
  )
