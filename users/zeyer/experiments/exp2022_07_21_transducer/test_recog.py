"""
Demo/test for recog
"""


from __future__ import annotations
from sisyphus import tk
from .task import get_nltk_timit_task
from .pipeline_swb_2020 import recog, model_recog, from_scratch_model_def
from .recog import ModelWithCheckpoint
from .model import Checkpoint


def _dummy_path_available(_) -> bool:
    return True


def sis_config_main():
    """sis config function"""
    task = get_nltk_timit_task()

    model = ModelWithCheckpoint(
        definition=from_scratch_model_def,
        checkpoint=Checkpoint(
            index_path=tk.Path("dummy-non-existing-model.ckpt-100000.index", available=_dummy_path_available)))

    tk.register_output('test_recog', recog(task, model, recog_def=model_recog).main_measure_value)


py = sis_config_main  # `py` is the default sis config function name
