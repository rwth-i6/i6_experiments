"""
Qwen2
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple, Dict, Any
import functools

from returnn.util.basic import prod
from sisyphus import Path

from i6_core.returnn.training import PtCheckpoint
from i6_experiments.common.utils.fake_job import make_fake_job
from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint, ModelDefWithCfg, ModelDef

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, single_step_dim


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

        # Directly put model.model here for compatible parameters.
        self.model = model.model
        # Don't put the model here, to not confuse the parameters.
        self._call_func = model.call_func
        self._embed_func = model.embed_func

    def default_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
        """Default initial state"""
        hist_dim = Dim(0, name="history")
        state = rf.State(
            batch_dims=list(batch_dims),
            pos=rf.constant(0, dims=batch_dims, dtype="int32"),
            hist_dim=hist_dim,
            mask=rf.constant(True, dims=batch_dims + [hist_dim], dtype="bool"),
            past_key_values=None,
            past_key_values_dims=None,
        )
        return state

    def __call__(self, source: Tensor, *, spatial_dim: Dim, state: rf.State) -> Tuple[Tensor, rf.State]:
        """
        forward, single step or whole sequence.

        :param source: labels
        :param spatial_dim: single_step_dim or spatial dim of source
        :param state: e.g. via :func:`default_initial_state`
        :return: logits, new state
        """
        from transformers.cache_utils import DynamicCache
        import tree
        import torch

        batch_dims = state.batch_dims
        merged_batch_dim: Dim = prod(batch_dims)
        spatial_dim_ = spatial_dim if spatial_dim != single_step_dim else Dim(1, name="time_single_step")
        source = source.copy_compatible_to_dims(batch_dims + [spatial_dim_], unbroadcast=True)
        new_pos = state.pos + spatial_dim_.get_size_tensor(device=state.pos.device)
        # We just keep all the padding. The explicit pos ids and masks handle the rest.
        hist_padded_dim = Dim(state.hist_dim.get_dim_value_tensor(), name="padded_history")
        new_mask, new_hist_dim = rf.concat(
            (rf.replace_dim(state.mask, in_dim=state.hist_dim, out_dim=hist_padded_dim)[0], hist_padded_dim),
            (rf.sequence_mask(spatial_dim_, device=state.mask.device), spatial_dim_),
        )

        def _combine_batch_and_beam(obj: Optional[Tensor]) -> Optional[Tensor]:
            if obj is None:
                return None
            assert isinstance(obj, Tensor), f"expected Tensor, got {obj} {type(obj)}"
            for dim in batch_dims:
                assert dim in obj.dims, f"expected {dim} in {obj.dims} for {obj}"
            obj, _ = rf.merge_dims(obj, dims=batch_dims, out_dim=merged_batch_dim)
            return obj.copy_compatible_to_dims(
                [merged_batch_dim] + [dim for dim in obj.dims if dim != merged_batch_dim], add_dims=False
            )

        def _combine_batch_and_beam_raw(obj: Optional[Tensor]) -> Optional[torch.Tensor]:
            if obj is None:
                return None
            assert isinstance(obj, Tensor), f"expected Tensor, got {obj} {type(obj)}"
            obj = _combine_batch_and_beam(obj)
            return obj.raw_tensor

        # dims must be equal for rf.nested.where_nested etc, thus reuse them
        dim_cache = {}

        def _make_dim_from_raw(obj_raw: torch.Tensor, i: int) -> Dim:
            raw_size = int(obj_raw.size(i))
            if obj_raw.dim() == 4 and i == 2:  # assume (batch*beam, num_heads, time, head_dim)
                assert raw_size == new_hist_dim.get_dim_value(), (
                    f"expected time dim {new_hist_dim.get_dim_value()=} at dim 2,"
                    f" got {raw_size=} for {obj_raw.shape=},"
                    f" {state.hist_dim=} {state.hist_dim.get_size_tensor().raw_tensor.numpy()=}"
                )
                return new_hist_dim
            cache_key = (i, raw_size)
            if cache_key in dim_cache:
                return dim_cache[cache_key]
            return dim_cache.setdefault(cache_key, Dim(raw_size, name=f"dim{i}_{raw_size}"))

        def _separate_batch_and_beam(obj_raw: torch.Tensor, *, dims: Optional[Sequence[Dim]] = None) -> Tensor:
            assert isinstance(obj_raw, torch.Tensor), f"expected torch.Tensor, got {obj_raw} {type(obj_raw)}"
            if dims is not None:
                assert dims[0] == merged_batch_dim
                assert len(dims) == obj_raw.dim()
                assert all(dims[i].get_dim_value() == obj_raw.size(i) for i in range(obj_raw.dim())), (
                    f"expected dims {[(d, d.get_dim_value()) for d in dims]} to match obj_raw {obj_raw.shape}"
                )
            else:
                assert merged_batch_dim.get_dim_value() == obj_raw.size(0)
                dims = [merged_batch_dim] + [_make_dim_from_raw(obj_raw, i) for i in range(1, obj_raw.dim())]
            obj = rf.convert_to_tensor(obj_raw, dims=dims)
            return rf.split_dims(obj, axis=merged_batch_dim, dims=batch_dims)

        source_raw = _combine_batch_and_beam_raw(source)
        input_embeds_raw = self._embed_func(source_raw)
        past_key_values_raw = tree.map_structure(_combine_batch_and_beam_raw, state.past_key_values)

        # See transformers.masking_utils.create_causal_mask for doc.
        # attention_mask: (batch_size, number_of_seen_tokens+q_length)
        attention_mask_raw = _combine_batch_and_beam_raw(
            rf.where(new_mask, 0.0, float("-inf")).copy_compatible_to_dims(
                batch_dims + [new_hist_dim], unbroadcast=True
            )
        )
        # position_ids: (batch_size, query_length)
        position_ids_raw = _combine_batch_and_beam_raw(
            rf.combine_bc(
                state.pos,
                "+",
                rf.range_over_dim(spatial_dim_, device=source.device),
            ).copy_compatible_to_dims(batch_dims + [spatial_dim_], unbroadcast=True)
        )
        output = self._call_func(
            past_key_values=DynamicCache.from_legacy_cache(past_key_values_raw),
            inputs_embeds=input_embeds_raw,
            attention_mask=attention_mask_raw,
            position_ids=position_ids_raw,
            use_cache=True,
            logits_to_keep=slice(None),
        )
        past_key_values_raw_ = output.past_key_values
        assert isinstance(past_key_values_raw_, DynamicCache)
        logits_raw = output.logits
        assert isinstance(logits_raw, torch.Tensor)
        assert logits_raw.shape[-1] >= self.vocab_dim.dimension  # it might be larger due to optimization
        logits_raw = logits_raw[..., : self.vocab_dim.dimension]  # (batch*beam, time, vocab)

        def _get_dims_from_tensor(tensor: Tensor) -> Sequence[Dim]:
            return tensor.dims

        new_state = rf.State(
            batch_dims=batch_dims,
            pos=new_pos,
            hist_dim=new_hist_dim,
            mask=new_mask,
            past_key_values=tree.map_structure(_separate_batch_and_beam, past_key_values_raw_.to_legacy_cache()),
        )
        new_state.past_key_values_dims = tree.map_structure(_get_dims_from_tensor, new_state.past_key_values)
        logits = _separate_batch_and_beam(
            logits_raw, dims=[merged_batch_dim, spatial_dim_, self.vocab_dim]
        )  # (batch, beam, time, vocab)
        if spatial_dim == single_step_dim:
            logits = rf.squeeze(logits, spatial_dim_)
        return logits, new_state
