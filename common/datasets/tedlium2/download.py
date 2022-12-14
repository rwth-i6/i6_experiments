import os
from functools import lru_cache
from typing import Any, Dict

from i6_core.datasets.tedlium2 import (
    DownloadTEDLIUM2CorpusJob,
    CreateTEDLIUM2BlissCorpusJob,
)


@lru_cache()
def download_data_dict(
    output_prefix: str = "datasets"
) -> Dict[str, Any]:
    data_dict = {}

    download_tedlium2_job = DownloadTEDLIUM2CorpusJob()
    download_tedlium2_job.add_alias(
        os.path.join(output_prefix, "download", "raw_corpus_job")
    )
    data_dict["data_dir"] = download_tedlium2_job.out_corpus_folders
    data_dict["lm_dir"] = download_tedlium2_job.out_lm_folder
    data_dict["vocab"] = download_tedlium2_job.out_vocab_dict

    bliss_corpus_tedlium2_job = CreateTEDLIUM2BlissCorpusJob(download_tedlium2_job.out_corpus_folders)
    bliss_corpus_tedlium2_job.add_alias(
        os.path.join(output_prefix, "create_bliss", "bliss_corpus_job")
    )
    data_dict["bliss_nist"] = bliss_corpus_tedlium2_job.out_corpus_files
    data_dict["stm"] = bliss_corpus_tedlium2_job.out_stm_files

    return data_dict
