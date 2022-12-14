import os
from functools import lru_cache
from typing import Dict
import pathlib

from sisyphus import tk

from .download import download_data_dict


@lru_cache()
def get_text_data_dict(output_prefix: str = "datasets") -> Dict[str, tk.Path]:
    txt_dict = {}

    lm_dir = download_data_dict(output_prefix=output_prefix)["lm_dir"]

    txt_list = os.listdir(lm_dir)

    for t in txt_list:
        name = os.path.splitext(os.path.basename(t))[0]
        path = tk.Path(os.path.abspath(t), cached=True)
        txt_dict[name] = path

    return txt_dict
