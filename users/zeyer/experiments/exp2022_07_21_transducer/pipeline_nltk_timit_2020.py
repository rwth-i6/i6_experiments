"""
2020 pipeline from switchboard applied to NLTK TIMIT (small subset of TIMIT, but freely available via NLTK)
"""


from __future__ import annotations
from i6_experiments.users.zeyer.datasets.nltk_timit import get_nltk_timit_task
from .pipeline_swb_2020 import pipeline


def sis_config_main():
    """sis config function"""
    task = get_nltk_timit_task()
    pipeline(task)


py = sis_config_main  # `py` is the default sis config function name
