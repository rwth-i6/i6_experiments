from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence, List, Collection
import tree
from itertools import product
import copy

from sisyphus import tk

from returnn.tensor import Tensor

from i6_core.returnn.training import PtCheckpoint
from i6_experiments.users.yang.torch.decoding.ctc_aed_joint_simple_version import model_recog

from i6_experiments.users.yang.torch.decoding.ctc_greedy import model_recog_greedy
from i6_experiments.users.yang.torch.decoding.ctc_label_sync import model_recog_label_sync
from i6_experiments.users.yang.torch.decoding.recog import recog_model
from i6_experiments.users.yang.torch.alignment.align import align_forward
from i6_experiments.users.yang.torch.alignment.ctc_alignment import _forced_align
from i6_experiments.users.yang.torch.luca_ctc.model_conformer_kd_ctc import from_scratch_model_def
from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint
from i6_experiments.users.yang.torch.albert_exp2024_04_23_baselines.aed_new import aed_model_def





_sis_prefix: Optional[str] = None

default_search_args = {
    "beam_size": 12,
}

def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

        prefix_name = get_setup_prefix_for_module(__name__)
    global _sis_prefix
    _sis_prefix = prefix_name


def get_model_ckpt(model_name, model_path, model_def=from_scratch_model_def):
    new_ckpt_path = tk.Path(model_path,
                            hash_overwrite=model_name + '_torch_ckpt')
    new_ckpt = PtCheckpoint(new_ckpt_path)
    model_ckpt = ModelWithCheckpoint(definition=model_def, checkpoint=new_ckpt)
    return model_ckpt





def sis_run_with_prefix(prefix_name: Optional[str] = None):

    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_raw_v2
    task = get_librispeech_task_raw_v2(vocab="bpe10k")
    if _sis_prefix is None:
        _sis_setup_global_prefix()
    model_names = []
    models_with_pt_ckpt = {}

    model_name = "ctc_baseline"
    # load checkpoints:
    # model trained by Luca, eos not used in training,
    # {"dev-clean": 3.02, "dev-other": 6.8, "test-clean": 3.16, "test-other": 7.07},
    # verify the decoding result: bsf=160: the same result
    model_path = "/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"
    new_ckpt_path = tk.Path(
        model_path,
        hash_overwrite=model_name + "_torch_ckpt",
    )
    new_ckpt = PtCheckpoint(new_ckpt_path)
    ctc_model_ckpt = ModelWithCheckpoint(definition=from_scratch_model_def, checkpoint=new_ckpt)
    # debug with dev-other

    search_args = {
        "bsf": 160,
        "batch_size_dependent": True,
    }
    model_args = {
        "ctc_output_args": {
            "ctc_enc_layer_id": 12,
        }
    }
    name = model_name + '-alignment'
    dataset = task.eval_datasets["dev-other"]
    align_result = align_forward(
        dataset=dataset,
        model=ctc_model_ckpt,
        align_def=_forced_align,
        model_args=model_args,
        search_args=search_args,
    )

    tk.register_output(
        _sis_prefix + '/' + name,
        align_result,
    )












py = sis_run_with_prefix


