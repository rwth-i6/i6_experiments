from functools import lru_cache
import os
from sisyphus import tk
from typing import Optional

from i6_core.returnn.oggzip import BlissToOggZipJob
from i6_core.datasets.loquacious import (
    PrepareLoquaciousTrainSmallDatasetJob,
    PrepareLoquaciousTrainMediumDatasetJob,
    PrepareLoquaciousTestDatasetsJob,
)


@lru_cache()
def get_bliss_corpus_dict(
    hf_home_dir=tk.Path("/work/common/asr/loquacious/huggingface"),
    output_prefix="datasets",
    test_dataset_version: int = 1,
    train_medium_dataset_version: int = 1,
):
    """
    Download and create a bliss corpus for each of the LibriSpeech training corpora and test sets,
    and return all corpora as a single corpus dict.

    Outputs will always be .ogg

    No outputs will be registered.

    :param str output_prefix:
    :param test_dataset_version: version of the dev/test preparation job
    :param train_medium_dataset_version: version of the medium training-set preparation job
    :return: A corpus dict with the following entries:
        - {dev|test}.all
        - {dev|test}.commonvoice
        - {dev|test}.librispeech
        - {dev|test}.voxpopuli
        - {dev|test}.yodas
        - train.small
        - train.medium
        - train.medium-wo-small

    :rtype: dict[str, tk.Path]
    """

    output_prefix = os.path.join(output_prefix, "Loquacious")

    bliss_corpus_dict = {}

    if test_dataset_version == 1:
        devtest_job = PrepareLoquaciousTestDatasetsJob(hf_home_dir=hf_home_dir)
        devtest_alias = "prepare_dev_test"
    elif test_dataset_version == 2:
        from i6_core.datasets.loquacious import PrepareLoquaciousTestDatasetsJobV2

        devtest_job = PrepareLoquaciousTestDatasetsJobV2(hf_home_dir=None)
        devtest_alias = "prepare_dev_test_v2"
    else:
        raise ValueError(f"Unsupported Loquacious test dataset version {test_dataset_version}")
    small_job = PrepareLoquaciousTrainSmallDatasetJob(hf_home_dir=hf_home_dir)
    if train_medium_dataset_version == 1:
        medium_job = PrepareLoquaciousTrainMediumDatasetJob(hf_home_dir=hf_home_dir)
        medium_alias = "prepare_medium"
    elif train_medium_dataset_version == 2:
        from i6_core.datasets.loquacious import PrepareLoquaciousTrainMediumDatasetJobV2

        medium_job = PrepareLoquaciousTrainMediumDatasetJobV2(hf_home_dir=None)
        medium_alias = "prepare_medium_v2"
    else:
        raise ValueError(f"Unsupported Loquacious medium dataset version {train_medium_dataset_version}")
    devtest_job.add_alias(output_prefix + f"/{devtest_alias}")
    small_job.add_alias(output_prefix + "/prepare_small")
    medium_job.add_alias(output_prefix + f"/{medium_alias}")

    for key, corpus in devtest_job.out_dev_corpora.items():
        bliss_corpus_dict[f"dev.{key}"] = corpus

    for key, corpus in devtest_job.out_test_corpora.items():
        bliss_corpus_dict[f"test.{key}"] = corpus

    bliss_corpus_dict["train.small"] = small_job.out_corpus
    bliss_corpus_dict["train.medium"] = medium_job.out_corpus
    bliss_corpus_dict["train.medium-wo-small"] = medium_job.out_corpus_wo_small

    return bliss_corpus_dict


@lru_cache()
def get_ogg_zip_dict(
    hf_home_dir=tk.Path("/work/common/asr/loquacious/huggingface"),
    output_prefix: str = "datasets",
    returnn_python_exe: Optional[tk.Path] = None,
    returnn_root: Optional[tk.Path] = None,
    test_dataset_version: int = 1,
    train_medium_dataset_version: int = 1,
):
    """
    Get a dictionary containing the paths to the ogg_zip for each corpus part.

    No outputs will be registered.

    :param str output_prefix:
    :param
    :return: dictionary with ogg zip paths for:
        - {dev|test}.all
        - {dev|test}.commonvoice
        - {dev|test}.librispeech
        - {dev|test}.voxpopuli
        - {dev|test}.yodas
        - train.small
        - train.medium
        - train.medium-wo-small
    :rtype: dict[str, tk.Path]
    """

    ogg_zip_dict = {}
    bliss_corpus_dict = get_bliss_corpus_dict(
        hf_home_dir=hf_home_dir,
        output_prefix=output_prefix,
        test_dataset_version=test_dataset_version,
        train_medium_dataset_version=train_medium_dataset_version,
    )
    for name, bliss_corpus in bliss_corpus_dict.items():
        ogg_zip_job = BlissToOggZipJob(
            bliss_corpus,
            no_conversion=True,
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
        )
        ogg_zip_job.add_alias(os.path.join(output_prefix, "Loquacious", "%s_ogg_zip_job" % name))
        ogg_zip_dict[name] = ogg_zip_job.out_ogg_zip

    return ogg_zip_dict
