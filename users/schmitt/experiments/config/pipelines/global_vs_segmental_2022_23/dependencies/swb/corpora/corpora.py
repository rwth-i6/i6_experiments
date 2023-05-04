from i6_core.corpus.segments import SegmentCorpusJob, ShuffleAndSplitSegmentsJob
from i6_core.corpus.convert import CorpusToStmJob
from i6_core.corpus.filter import FilterCorpusBySegmentsJob

from sisyphus import *


class SWBCorpora:
  corpus_paths = {
    "train": Path("/work/asr3/irie/data/switchboard/corpora/train.corpus.gz", cached=True),
    "dev": Path("/work/asr3/irie/data/switchboard/corpora/dev.corpus.gz", cached=True),
    "hub5e_01": Path("/work/asr3/irie/data/switchboard/corpora/hub5e_01.corpus.gz", cached=True),
    "rt03s": Path("/work/asr3/irie/data/switchboard/corpora/rt03s.corpus.gz", cached=True)}
  corpus_paths["devtrain"] = corpus_paths["train"]
  corpus_paths["cv"] = corpus_paths["train"]
  corpus_paths["dev400"] = corpus_paths["dev"]
  corpus_paths["cv300"] = corpus_paths["train"]
  corpus_paths["cv_test"] = corpus_paths["train"]

  feature_cache_paths = {
    "train": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.train.bundle", cached=True),
    "dev": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.dev.bundle", cached=True),
    "hub5e_01": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.hub5e_01.bundle", cached=True),
    "rt03s": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.rt03s.bundle", cached=True)}
  feature_cache_paths["devtrain"] = feature_cache_paths["train"]
  feature_cache_paths["cv"] = feature_cache_paths["train"]
  feature_cache_paths["cv300"] = feature_cache_paths["train"]
  feature_cache_paths["cv_test"] = feature_cache_paths["train"]
  feature_cache_paths["dev400"] = feature_cache_paths["dev"]

  raw_audio_paths = {
    "train": Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.tSpSJCnE1d2G/output/out.ogg.zip", cached=True),
    "dev": Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.tSpSJCnE1d2G/output/out.ogg.zip", cached=True),
    "hub5e_01": Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.tSpSJCnE1d2G/output/out.ogg.zip", cached=True),
    "rt03s": Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.tSpSJCnE1d2G/output/out.ogg.zip", cached=True),
  }
  raw_audio_paths["devtrain"] = raw_audio_paths["train"]
  raw_audio_paths["cv"] = raw_audio_paths["train"]
  raw_audio_paths["cv300"] = raw_audio_paths["train"]
  raw_audio_paths["cv_test"] = raw_audio_paths["train"]
  raw_audio_paths["dev400"] = raw_audio_paths["dev"]

  segment_paths = {
    "dev": SegmentCorpusJob(corpus_paths["dev"], 1).out_single_segment_files[1],
    "hub5e_01": SegmentCorpusJob(corpus_paths["hub5e_01"], 1).out_single_segment_files[1],
    "rt03s": SegmentCorpusJob(corpus_paths["rt03s"], 1).out_single_segment_files[1],
    "cv_test": Path("/work/asr3/zeyer/schmitt/tests/swb1/bpe-transducer_decoding-test/cv_test_segments1")
  }
  segment_paths["dev400"] = ShuffleAndSplitSegmentsJob(
    segment_file=segment_paths["dev"], split={"10": 0.1, "90": 0.9}).out_segments["10"]

  test_stm_paths = {
    "dev": Path("/u/tuske/bin/switchboard/hub5e_00.2.stm"),
    "hub5e_01": Path("/u/tuske/bin/switchboard/hub5e_01.2.stm"),
    "rt03s": Path("/u/tuske/bin/switchboard/rt03s_ctsonly.stm"),
    "dev400": CorpusToStmJob(
      bliss_corpus=FilterCorpusBySegmentsJob(
        bliss_corpus=corpus_paths["dev"],
        segment_file=segment_paths["dev400"]).out_corpus).out_stm_path}

  corpus_keys = ("train", "cv", "cv300", "cv_test", "devtrain", "dev", "dev400", "hub5e_01", "rt03s")
  train_corpus_keys = ("train", "cv", "devtrain")
  test_corpus_keys = ("dev", "dev400", "hub5e_01", "rt03s", "cv", "cv300", "cv_test")
