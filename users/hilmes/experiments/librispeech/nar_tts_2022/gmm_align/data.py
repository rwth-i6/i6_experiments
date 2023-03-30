"""
Defines the data inputs for any RASR based LibriSpeech task
"""
from dataclasses import dataclass
from typing import Dict
from i6_core.corpus.filter import FilterCorpusBySegmentsJob
from i6_core.meta import CorpusObject
from sisyphus import tk

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
from i6_experiments.users.hilmes.tools.tts.speaker_embeddings import RemoveSpeakerTagsJob


@dataclass()
class CorpusData:
  """
    Helper class to define all RasrDataInputs to be passed to the `System` class
    """

  train_data: Dict[str, RasrDataInput]
  dev_data: Dict[str, RasrDataInput]
  test_data: Dict[str, RasrDataInput]


def get_corpus_data_inputs(remove_speakers=False):
  """
    :return: a 3-sized tuple containing lists of RasrDataInput for train, dev and test
    """

  # this is the FFmpeg silence preprocessed version of LibriSpeech train-clean-100
  sil_pp_train_clean_100 = get_ls_train_clean_100_tts_silencepreprocessed()
  if remove_speakers:
    remove_job = RemoveSpeakerTagsJob(sil_pp_train_clean_100.corpus_file)
    sil_pp_train_clean_100.corpus_file = remove_job.out_corpus
    tk.register_output("corpora/no_speaker/corpus.xml.gz", remove_job.out_corpus)

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

  corpus_object_dict = get_corpus_object_dict(audio_format="wav", output_prefix="corpora")

  for dev_key in ["dev-clean", "dev-other"]:
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


def get_corpus_data_inputs_no_sil_p():
  """
    :return: a 3-sized tuple containing lists of RasrDataInput for train, dev and test
    """

  # this is the FFmpeg silence preprocessed version of LibriSpeech train-clean-100
  corpus_object_dict = get_corpus_object_dict(audio_format="ogg", output_prefix="corpora")
  sil_pp_train_clean_100 = corpus_object_dict["train-clean-100"]
  sil_pp_train_clean_100 = deepcopy(sil_pp_train_clean_100)
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

  corpus_object_dict = get_corpus_object_dict(audio_format="wav", output_prefix="corpora")

  for dev_key in ["dev-clean", "dev-other"]:
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


def get_synth_corpus_data_inputs(synth_corpus: tk.Path, ls360=False, ls860=False, ls1000=False):
  """
    :return: a 3-sized tuple containing lists of RasrDataInput for train, dev and test
    """

  # segments for train-clean-100-tts-train and train-clean-100-tts-dev
  # (1004 segments for dev, 4 segments for each of the 251 speakers)
  sil_pp_train_clean_100 = get_ls_train_clean_100_tts_silencepreprocessed()
  synth_co = CorpusObject()
  synth_co.audio_format = "ogg"
  synth_co.corpus_file = synth_corpus
  synth_co.duration = sil_pp_train_clean_100.duration
  # remove the dev segments from the GMM training corpus

  g2p_lexica = get_g2p_augmented_bliss_lexicon_dict(
    output_prefix="corpora",
    add_unknown_phoneme_and_mapping=False,
    use_stress_marker=False,
  )
  lm = None
  if ls860:
    file = g2p_lexica["train-other-960"]
  elif ls360:
    file = g2p_lexica["train-clean-360"]
  else:
    file = g2p_lexica["train-clean-100"]

  train_lexicon = {
    "filename": file,
    "normalize_pronunciation": False,
  }

  train_data_inputs = {}
  dev_data_inputs = {}
  test_data_inputs = {}

  train_data_inputs["train-clean-100"] = RasrDataInput(
    corpus_object=synth_co,
    concurrent=constants.concurrent["train-clean-100"] if not (ls860 or ls1000) else constants.concurrent["train-clean-100"] * 10,
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

  corpus_object_dict = get_corpus_object_dict(audio_format="wav", output_prefix="corpora")

  for dev_key in ["dev-clean", "dev-other"]:
    dev_data_inputs[dev_key] = RasrDataInput(
      corpus_object=corpus_object_dict[dev_key],
      concurrent=constants.concurrent[dev_key],
      lexicon=lexicon,
      lm=lm,
    )

  # for test_key in ["test-clean", "test-other"]:
  #    test_data_inputs[test_key] = RasrDataInput(
  #        corpus_object=corpus_object_dict[test_key],
  #        concurrent=constants.concurrent[test_key],
  #        lexicon=lexicon,
  #        lm=lm,
  #    )

  test_data_inputs["tts_align"] = RasrDataInput(
    corpus_object=synth_co,
    concurrent=constants.concurrent["train-clean-100"] if not (ls860 or ls1000) else constants.concurrent["train-clean-100"] * 10,
    lexicon=train_lexicon,
    lm=lm,
  )

  return CorpusData(
    train_data=train_data_inputs,
    dev_data=dev_data_inputs,
    test_data=test_data_inputs,
  )


def get_sil_corpus_data_inputs():
  """
    :return: a 3-sized tuple containing lists of RasrDataInput for train, dev and test
    """

  # this is the FFmpeg silence preprocessed version of LibriSpeech train-clean-100
  sil_pp_train_clean_100 = get_ls_train_clean_100_tts_silencepreprocessed()

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

  corpus_object_dict = get_corpus_object_dict(audio_format="wav", output_prefix="corpora")

  for dev_key in ["dev-clean", "dev-other"]:
    dev_data_inputs[dev_key] = RasrDataInput(
      corpus_object=corpus_object_dict[dev_key],
      concurrent=constants.concurrent[dev_key],
      lexicon=lexicon,
      lm=lm,
    )

  test_data_inputs["tts_align"] = RasrDataInput(
    corpus_object=sil_pp_train_clean_100,
    concurrent=constants.concurrent["train-clean-100"],
    lexicon=train_lexicon,
    lm=lm,
  )

  return CorpusData(
    train_data=train_data_inputs,
    dev_data=dev_data_inputs,
    test_data=test_data_inputs,
  )


def get_corpus_data_inputs_no_speaker():
  """
    :return: a 3-sized tuple containing lists of RasrDataInput for train, dev and test
    """

  # this is the FFmpeg silence preprocessed version of LibriSpeech train-clean-100
  corpus_object_dict = get_corpus_object_dict(audio_format="ogg", output_prefix="corpora")
  sil_pp_train_clean_100 = corpus_object_dict["train-clean-100"]
  sil_pp_train_clean_100 = deepcopy(sil_pp_train_clean_100)

  remove_job = RemoveSpeakerTagsJob(sil_pp_train_clean_100.corpus_file)
  sil_pp_train_clean_100.corpus_file = remove_job.out_corpus
  tk.register_output("corpora/no_speaker/corpus.xml.gz", remove_job.out_corpus)

  sil_pp_train_clean_100_align = deepcopy(sil_pp_train_clean_100)

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

  corpus_object_dict = get_corpus_object_dict(audio_format="wav", output_prefix="corpora")

  for dev_key in ["dev-clean", "dev-other"]:
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
