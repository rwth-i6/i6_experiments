from typing import Union, Dict
import scipy
import numpy as np
import torch

from i6_experiments.users.phan.models.multi_model_wrapper import (
    MultiModelWrapper, MultiModelWrapperConfig,
    wrapper_config_import_obj, get_base_serializer,
)
from i6_experiments.users.phan.models.trainable_unigram import Unigram
from i6_experiments.users.phan.utils import write_log_probs_to_files
from i6_experiments.common.setups.returnn_pytorch.serialization import (
    Collection, Call,
)
from i6_experiments.common.setups.serialization import (
    PartialImport,
)
from i6_core.returnn.config import CodeWrapper

def get_prior_from_labelwise_unigram(
    cfg: MultiModelWrapperConfig,
    model_path: str,
    student_lm_key: str = "student_lm",
    blank_idx: int = 0,
    log_prob_blank: float = 0.,
):
    """
    The prior is renormalized among true labels. For blank,
    a default value is assigned, mostly 1.
    """
    model = MultiModelWrapper(cfg=cfg)
    model.load_state_dict(torch.load(model_path)["model"])
    if isinstance(model, MultiModelWrapper):
        model = model.module_dict[student_lm_key]
    assert isinstance(model, Unigram), "Here the model should be a trainable Unigram"
    log_probs = model.unigram_tensor.log_softmax(dim=-1).detach().cpu().numpy()
    log_probs[blank_idx] = log_prob_blank
    out_idx_wo_blank = np.arange(len(log_probs)) != blank_idx
    log_probs_wo_blank = log_probs[out_idx_wo_blank]
    log_probs[out_idx_wo_blank] = log_probs_wo_blank - scipy.special.logsumexp(log_probs_wo_blank)
    write_log_probs_to_files(log_probs)


def get_serializer_labelwise_unigram(
    action_name: str,
    cfg: MultiModelWrapperConfig,
    student_lm_key: str = "student_lm",
    blank_idx: int = 0,
    log_prob_blank: float = 0.,
    config_variable_name: str = "cfg",
) -> Collection:
    """
    Task in returnn config shoulf be config:<action_name>.
    You have to serialize the cfg yourself
    """
    action_import = PartialImport(
        code_object_path=f"{__name__}.{get_prior_from_labelwise_unigram.__name__}",
        unhashed_package_root=f"{__name__}",
        hashed_arguments={
            "cfg": CodeWrapper(config_variable_name),
            "model_path": CodeWrapper("load"),
            "student_lm_key": student_lm_key,
            "blank_idx": blank_idx,
            "log_prob_blank": log_prob_blank,
        },
        unhashed_arguments={},
        import_as=action_name,
    )
    return action_import
