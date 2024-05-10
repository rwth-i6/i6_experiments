from sisyphus import *

from i6_core.corpus.convert import CorpusToStmJob

from i6_experiments.common.datasets.tedlium2.corpus import get_bliss_corpus_dict


class TedLium2Corpora:
  def __init__(self):
    self.oggzip_paths = {
      "train": [Path("/work/asr4/zeineldeen/setups-data/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.kjOfflyMvuHh/output/out.ogg.zip")],
      "cv": [],
      "dev": [Path("/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.Wgp6724p1XD2/output/out.ogg.zip")],
      "test": [Path("/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.c1QY7FXTR0Ef/output/out.ogg.zip")]
    }
    self.oggzip_paths["devtrain"] = self.oggzip_paths["train"]

    self.corpus_paths = get_bliss_corpus_dict(audio_format="wav")

    self.segment_paths = {
      "train": None,
      "cv": None
    }

    self.stm_jobs = {
      corpus_key: CorpusToStmJob(
        bliss_corpus=self.corpus_paths[corpus_key],
      ) for corpus_key in ("dev", "test")
    }
    self.stm_paths = {corpus_key: stm_job.out_stm_path for corpus_key, stm_job in self.stm_jobs.items()}

    self.corpus_keys = ("train", "cv", "devtrain")
    self.train_corpus_keys = ("train", "cv", "devtrain")

    self.partition_epoch = None
