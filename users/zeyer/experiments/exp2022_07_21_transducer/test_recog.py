"""
Demo/test for recog
"""


from __future__ import annotations
from sisyphus import tk
from i6_experiments.users.zeyer.datasets.nltk_timit import get_nltk_timit_task
from .pipeline_swb_2020 import model_recog, from_scratch_model_def
from i6_experiments.users.zeyer.returnn.training import ReturnnInitModelJob
from i6_experiments.users.zeyer.recog import ModelWithCheckpoint, search_config, recog_model


def sis_config_main():
    """sis config function"""
    task = get_nltk_timit_task()
    model_def = from_scratch_model_def
    recog_def = model_recog

    config = search_config(task.dev_dataset, model_def, recog_def)
    config.config.pop("dev")
    checkpoint = ReturnnInitModelJob(config).out_checkpoint
    model = ModelWithCheckpoint(definition=model_def, checkpoint=checkpoint)

    tk.register_output('test_recog', recog_model(task, model, recog_def=recog_def).main_measure_value)


py = sis_config_main  # `py` is the default sis config function name
