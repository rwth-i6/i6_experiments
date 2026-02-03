"""
Qwen2
"""

from __future__ import annotations
from typing import Sequence, Tuple, Dict, Any
import functools

from sisyphus import Path

from i6_core.returnn.training import PtCheckpoint
from i6_experiments.common.utils.fake_job import make_fake_job
from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint, ModelDefWithCfg, ModelDef

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim


def get_lm() -> ModelWithCheckpoint:
    """
    Keep compat to :mod:`i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.ctc_delayed_fusion_v2`.
    """

    # /hpcwork/p0023999/hq237549/sisyphus-work-dirs/2026-01-20--llm/work/i6_core/returnn/training/ReturnnTrainingJob.MIU24HbRi60L/output/returnn.config

    # noinspection PyTypeChecker
    get_model = functools.partial(
        Qwen2Model,
        **{
            # "/home/hq237549/experiments/2026-01-20--llm/work/i6_experiments/users/schmitt/external_models/huggingface/DownloadHuggingFaceRepoJob.r7AjtV7muFpk/output/hub_cache",
            "hf_hub_cache_dir": Path(
                "hub_cache",
                creator=make_fake_job(
                    module="i6_experiments.users.schmitt.external_models.huggingface",
                    name="DownloadHuggingFaceRepoJob",
                    sis_hash="r7AjtV7muFpk",
                ),
            ),
            "freeze_params": False,
            "lora_opts": None,
            "freeze_embedding_layer": True,
            "vocab_dim": {
                "name": "qwen_vocab",
                "dimension": 151646,
                "vocab": {
                    "class": "HuggingFaceTokenizer",
                    # "/home/hq237549/experiments/2026-01-20--llm/work/i6_experiments/users/schmitt/external_models/huggingface/DownloadHuggingFaceRepoJobV2.PUGzhO2dOEpK/output/content",
                    "huggingface_repo_dir": Path(
                        "content",
                        creator=make_fake_job(
                            module="i6_experiments.users.schmitt.external_models.huggingface",
                            name="DownloadHuggingFaceRepoJobV2",
                            sis_hash="PUGzhO2dOEpK",
                        ),
                    ),
                },
            },
        },
    )
    config = {}

    from speech_llm.prefix_lm.model.custom_missing_load_funcs.qwen import qwen_load_tied_embedding_matrices

    config["preload_from_files"] = {
        "qwen": {
            # /home/hq237549/experiments/2026-01-20--llm/work/i6_core/tools/download/DownloadJob.6SV1LOlUtQMG/output/qwen2-0_5b_model.safetensors
            "filename": Path(
                "qwen2-0_5b_model.safetensors",
                creator=make_fake_job(module="i6_core.tools.download", name="DownloadJob", sis_hash="6SV1LOlUtQMG"),
            ),
            "init_for_train": True,
            "checkpoint_key": None,
            "prefix": "model.",
            "custom_missing_load_func": qwen_load_tied_embedding_matrices,
            "ignore_missing": True,
        }
    }

    # ft_qwen0_5b_v2_bs25k_epoch100_part50_wup2.5_maxlr5e-06_frz_emb_full_ft--best            14.68     67.44:
    # /hpcwork/p0023999/hq237549/sisyphus-work-dirs/2026-01-20--llm/work/i6_core/returnn/training/GetBestPtCheckpointJob.biueEBxdJI4u/output/checkpoint.pt
    # from /hpcwork/p0023999/hq237549/sisyphus-work-dirs/2026-01-20--llm/work/i6_core/returnn/training/ReturnnTrainingJob.4mQ2qrd6hR3a

    # ft_qwen0_5b_v2_bs25k_epoch100_part50_wup2.5_maxlr5e-06_full_ft--best            14.39 65.40
    # /rwthfs/rz/cluster/hpcwork/p0023999/hq237549/sisyphus-work-dirs/2026-01-20--llm/work/i6_core/returnn/training/GetBestPtCheckpointJob.xUyBGR1LEpaY/output/checkpoint.pt

    checkpoint = Path(
        "checkpoint.pt",
        creator=make_fake_job(
            module="i6_core.returnn.training",
            name="GetBestPtCheckpointJob",
            sis_hash="xUyBGR1LEpaY",
        ),
    )

    get_model: ModelDef  # make compat
    get_model.behavior_version = 24
    get_model.backend = "torch"
    get_model.batch_size_factor = 1
    model_with_cfg = ModelDefWithCfg(model_def=get_model, config=config)

    return ModelWithCheckpoint(definition=model_with_cfg, checkpoint=PtCheckpoint(checkpoint))


class Qwen2Model(rf.Module):
    """
    Wraps Qwen2DecoderV3.

    Keep API compatible to how other RF LMs are expected, i.e. like RF TransformerDecoder.
    """

    def __init__(self, *, vocab_dim: Dict[str, Any], **kwargs):
        super().__init__()

        self.vocab_dim = Dim(**vocab_dim)

        from speech_llm.prefix_lm.model.definitions.decoders.qwen import Qwen2DecoderV3

        model = Qwen2DecoderV3(**kwargs)

        # The order matters for the parameter names.
        self.model = model.model
        self._model = model

    def default_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
        """Default initial state"""
        state = rf.State(
            s=self.s.default_initial_state(batch_dims=batch_dims),
            att=rf.zeros(list(batch_dims) + [self.att_num_heads * self.encoder_dim]),
            # Note: enc_spatial_dim missing! (Unfortunately the API of default_initial_state does not provide it.)
            accum_att_weights=rf.zeros(list(batch_dims) + [self.att_num_heads], feature_dim=self.att_num_heads),
        )
        state.att.feature_dim_axis = len(state.att.dims) - 1
        return state

    def __call__(self, source: Tensor, *, spatial_dim: Dim, state: rf.State) -> Tuple[Tensor, rf.State]:
        """
        forward, single step or whole sequence.

        :param source: labels
        :param spatial_dim: single_step_dim or spatial dim of source
        :param state: e.g. via :func:`default_initial_state`
        :return: logits, new state
        """
