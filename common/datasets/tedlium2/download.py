import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict

from sisyphus import tk

from i6_core.datasets.tedlium2 import (
    DownloadTEDLIUM2CorpusJob,
    CreateTEDLIUM2BlissCorpusJob,
)


@dataclass(frozen=True)
class TedLium2Data:
    """Class for storing the TedLium2 data"""

    data_dir: Dict[str, tk.Path]
    lm_dir: tk.Path
    vocab: tk.Path
    bliss_nist: Dict[str, tk.Path]
    stm: Dict[str, tk.Path]


@lru_cache()
def download_data_dict(output_prefix: str = "datasets") -> TedLium2Data:
    """
    downloads the TedLiumV2 dataset and performs the initial data processing steps

    :param output_prefix:
    :return:
    """
    download_tedlium2_job = DownloadTEDLIUM2CorpusJob()
    download_tedlium2_job.add_alias(os.path.join(output_prefix, "download", "raw_corpus_job"))

    bliss_corpus_tedlium2_job = CreateTEDLIUM2BlissCorpusJob(download_tedlium2_job.out_corpus_folders)
    bliss_corpus_tedlium2_job.add_alias(os.path.join(output_prefix, "create_bliss", "bliss_corpus_job"))

    bliss_unnormalized_job = CreateTEDLIUM2BlissCorpusJob(download_tedlium2_job.out_corpus_folders, normalize_stms=False)
    bliss_corpus_tedlium2_job.add_alias(os.path.join(output_prefix, "create_bliss", "bliss_corpus_job_unnorm"))
    tk.register_output("test/unnorm_dev", bliss_unnormalized_job.out_stm_files["dev"])

    tl2_data = TedLium2Data(
        data_dir=download_tedlium2_job.out_corpus_folders,
        lm_dir=download_tedlium2_job.out_lm_folder,
        vocab=download_tedlium2_job.out_vocab_dict,
        bliss_nist=bliss_corpus_tedlium2_job.out_corpus_files,
        stm=bliss_corpus_tedlium2_job.out_stm_files,
    )

    return tl2_data
