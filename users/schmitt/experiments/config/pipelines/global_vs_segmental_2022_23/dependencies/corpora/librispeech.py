from sisyphus import *

from i6_core.corpus.convert import CorpusToStmJob

from i6_experiments.common.datasets.librispeech.corpus import get_bliss_corpus_dict


class LibrispeechCorpora:
  oggzip_paths = {
    "train": [Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.Cbboscd6En6A/output/out.ogg.zip", cached=True)],
    "cv": [
      Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.RvwLniNrgMit/output/out.ogg.zip", cached=True),
      Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.NSdIHfk1iw2M/output/out.ogg.zip", cached=True)
    ],
    "dev-other": [Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.NSdIHfk1iw2M/output/out.ogg.zip", cached=True)]
  }
  oggzip_paths["devtrain"] = oggzip_paths["train"]

  segment_paths = {
    "train": None,
    "cv": Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/processing/PipelineJob.gTty7UHs0uBu/output/out", cached=True),
    "devtrain": None,
    "dev-other": None
  }

  bliss_corpus_dict = get_bliss_corpus_dict()

  stm_jobs = {
    "dev-other": CorpusToStmJob(bliss_corpus_dict["dev-other"])
  }

  stm_paths = {
    "dev-other": stm_jobs["dev-other"].out_stm_path
  }

  corpus_keys = ("train", "cv", "devtrain")
  train_corpus_keys = ("train", "cv", "devtrain")
  test_corpus_keys = ()
