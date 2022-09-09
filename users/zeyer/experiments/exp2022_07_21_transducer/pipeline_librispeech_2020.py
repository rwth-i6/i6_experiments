"""
2020 pipeline from switchboard applied to Librispeech
"""


from __future__ import annotations
from .task import get_librispeech_task
from .pipeline_swb_2020 import pipeline


def sis_config_main():
    """sis config function"""
    task = get_librispeech_task()
    pipeline(task)


py = sis_config_main  # `py` is the default sis config function name
