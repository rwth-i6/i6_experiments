"""
Based on the original config which we want to reproduce here, namely:
rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.config.py.
"""


from __future__ import annotations
from typing import Optional, Dict, Sequence
import contextlib
from returnn_common import nn
from returnn_common.nn.encoder.blstm_cnn_specaug import BlstmCnnSpecAugEncoder

from i6_experiments.users.zeyer.datasets.switchboard_2020.task import get_switchboard_task_bpe1k
from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
from i6_experiments.users.zeyer.recog import recog_training_exp
from ..train import train


def TODO_sis_run_with_prefix(prefix_name: str):
    """run the exp"""
    task = get_switchboard_task_bpe1k()
    model = train(task=task, config=config, model_def=from_scratch_model_def, train_def=from_scratch_training)
    recog_training_exp(prefix_name, task, model, recog_def=model_recog)

