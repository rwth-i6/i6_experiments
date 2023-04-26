from functools import lru_cache
from typing import Dict

from sisyphus import tk

from .download import download_data_dict


@lru_cache()
def get_text_data_dict(output_prefix: str = "datasets") -> Dict[str, tk.Path]:
    """
    gather all the textual data provided within the TedLiumV2 dataset

    :param output_prefix:
    :return:
    """
    lm_dir = download_data_dict(output_prefix=output_prefix).lm_dir

    text_corpora = [
        "commoncrawl-9pc",
        "europarl-v7-6pc",
        "giga-fren-4pc",
        "news-18pc",
        "news-commentary-v8-9pc",
        "yandex-1m-31pc",
    ]

    txt_dict = {name: lm_dir.join_right("%s.en.gz" % name) for name in text_corpora}

    return txt_dict
