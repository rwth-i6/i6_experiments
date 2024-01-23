import os.path

from i6_core.corpus.segments import SegmentCorpusJob, ShuffleAndSplitSegmentsJob
from i6_core.corpus.convert import CorpusToStmJob
from i6_core.corpus.filter import FilterCorpusBySegmentsJob
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.config import RasrConfigBuilder

from abc import ABC

from sisyphus import *


class SWBCorpus:
  def __init__(self):
    self.corpus_keys = ("train", "cv", "devtrain", "dev", "hub5e_01", "rt03s")

    self.stm_paths = {
      "dev": Path("/u/tuske/bin/switchboard/hub5e_00.2.stm"),
      "hub5e_01": Path("/u/tuske/bin/switchboard/hub5e_01.2.stm"),
      "rt03s": Path("/u/tuske/bin/switchboard/rt03s_ctsonly.stm"),
    }

    self.corpus_paths = {
      "train": Path("/work/asr3/irie/data/switchboard/corpora/train.corpus.gz", cached=True),
      "dev": Path("/work/asr3/irie/data/switchboard/corpora/dev.corpus.gz", cached=True),
      "hub5e_01": Path("/work/asr3/irie/data/switchboard/corpora/hub5e_01.corpus.gz", cached=True),
      "rt03s": Path("/work/asr3/irie/data/switchboard/corpora/rt03s.corpus.gz", cached=True)}
    self.corpus_paths["devtrain"] = self.corpus_paths["train"]
    self.corpus_paths["cv"] = self.corpus_paths["train"]

    self.feature_cache_paths = {
      "train": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.train.bundle", cached=True),
      "dev": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.dev.bundle", cached=True),
      "hub5e_01": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.hub5e_01.bundle", cached=True),
      "rt03s": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.rt03s.bundle", cached=True)}
    self.feature_cache_paths["devtrain"] = self.feature_cache_paths["train"]
    self.feature_cache_paths["cv"] = self.feature_cache_paths["train"]

    self.oggzip_paths = {
      "train": [Path(
        "/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.tSpSJCnE1d2G/output/out.ogg.zip",
        cached=True)],
      "cv": [Path(
        "/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.wjjrGz1EDF9t/output/out.ogg.zip",
        cached=True)],
      "dev": None
    }
    self.oggzip_paths["devtrain"] = self.oggzip_paths["train"]

    self.partition_epoch = 6

    self.rasr_feature_extraction_config_paths = {
      corpus_key: RasrConfigBuilder.get_feature_extraction_config(
        segment_path=None,
        feature_cache_path=self.feature_cache_paths[corpus_key],
        corpus_path=self.corpus_paths[corpus_key]) for corpus_key in self.corpus_keys}

class SWBCorpora(ABC):
  stm_paths = {
    "dev": Path("/u/tuske/bin/switchboard/hub5e_00.2.stm"),
    "hub5e_01": Path("/u/tuske/bin/switchboard/hub5e_01.2.stm"),
    "rt03s": Path("/u/tuske/bin/switchboard/rt03s_ctsonly.stm"),
  }

  corpus_paths = {
    "train": Path("/work/asr3/irie/data/switchboard/corpora/train.corpus.gz", cached=True),
    "dev": Path("/work/asr3/irie/data/switchboard/corpora/dev.corpus.gz", cached=True),
    "hub5e_01": Path("/work/asr3/irie/data/switchboard/corpora/hub5e_01.corpus.gz", cached=True),
    "rt03s": Path("/work/asr3/irie/data/switchboard/corpora/rt03s.corpus.gz", cached=True)}
  corpus_paths["devtrain"] = corpus_paths["train"]
  corpus_paths["cv"] = corpus_paths["train"]


class SWBSprintCorpora:
  corpus_paths = dict(**SWBCorpora.corpus_paths)
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

  segment_paths = {
    "dev": SegmentCorpusJob(corpus_paths["dev"], 1).out_single_segment_files[1],
    "hub5e_01": SegmentCorpusJob(corpus_paths["hub5e_01"], 1).out_single_segment_files[1],
    "rt03s": SegmentCorpusJob(corpus_paths["rt03s"], 1).out_single_segment_files[1],
    "cv_test": Path("/work/asr3/zeyer/schmitt/tests/swb1/bpe-transducer_decoding-test/cv_test_segments1")
  }
  segment_paths["dev400"] = ShuffleAndSplitSegmentsJob(
    segment_file=segment_paths["dev"], split={"10": 0.1, "90": 0.9}).out_segments["10"]

  # this is the base directory for the training segment files after excluding the seqs for which realignment fails
  realignment_train_segment_paths_dir = os.path.join(os.path.dirname(__file__), "segment_files")

  test_stm_paths = {
    **SWBCorpora.stm_paths,
    "dev400": CorpusToStmJob(
      bliss_corpus=FilterCorpusBySegmentsJob(
        bliss_corpus=corpus_paths["dev"],
        segment_file=segment_paths["dev400"]).out_corpus).out_stm_path}

  corpus_keys = ("train", "cv", "cv300", "cv_test", "devtrain", "dev", "dev400", "hub5e_01", "rt03s")
  train_corpus_keys = ("train", "cv", "devtrain")
  test_corpus_keys = ("dev", "dev400", "hub5e_01", "rt03s", "cv", "cv300", "cv_test")


class SWBOggZipCorpora:
  oggzip_paths = {
    "train": [Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.tSpSJCnE1d2G/output/out.ogg.zip", cached=True)],
    "cv": [Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.wjjrGz1EDF9t/output/out.ogg.zip", cached=True)],
    "dev": None
  }
  oggzip_paths["devtrain"] = oggzip_paths["train"]

  oggzip_bpe_codes = Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.FLNETa4J87YO/output/bpe.codes")
  oggzip_bpe_vocab = Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.FLNETa4J87YO/output/bpe.vocab")

  segment_paths = {
    "train": None,
    "cv": None,
    "devtrain": None,
    "dev": None
  }

  stm_paths = dict(**SWBCorpora.stm_paths)
  stm_jobs = {
    corpus_key: CorpusToStmJob(SWBCorpora.corpus_paths[corpus_key]) for corpus_key in ("dev", "hub5e_01", "rt03s")}

  corpus_keys = ("train", "cv", "devtrain", "dev", "hub5e_01", "rt03s")
  train_corpus_keys = ("train", "cv", "devtrain")
  test_corpus_keys = ("dev", "hub5e_01", "rt03s", "cv")
