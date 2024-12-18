"""
CTC recognition with LM
"""

from typing import Optional, Union, Any, TypeVar, Sequence, Tuple, Dict
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.decoder.transformer import TransformerDecoder

from sisyphus import tk

from returnn_common.datasets_old_2022_10.interface import DatasetConfig
from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, RecogDef, ModelWithCheckpoint

from i6_experiments.users.zeyer.recog import recog_model
from i6_experiments.users.zeyer.collect_model_dataset_stats import collect_statistics

from .ctc import Model, _get_ctc_model_kwargs_from_global_config, _batch_size_factor


_ctc_model_name = (
    "v6-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2"
    "-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001"
)

# trafo-n32-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b32_1k*: 34.03  -- still running...
# trafo-n96-d512-gelu-drop0-b32_1k: 34.96
# trafo-n24-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b100_5k-ep40: 35.60
# ...
# trafo-n24-d512-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b100_5k
# _lm_name = "trafo-n96-d512-gelu-drop0-b32_1k"
_lms = {
    "n24-d512": "trafo-n24-d512-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b100_5k",
    # "n96-d512": "trafo-n96-d512-gelu-drop0-b32_1k",
}


def py():
    """Sis entry point"""
    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_raw_v2

    if _sis_prefix is None:
        _sis_setup_global_prefix()

    prefix = f"{_sis_prefix}/ctc+lm"

    vocab = "spm10k"
    task = get_librispeech_task_raw_v2(vocab=vocab)
    ctc_model = _get_ctc_model(_ctc_model_name)
    prior = get_ctc_prior_probs(ctc_model, task.train_dataset.copy_train_as_static())
    tk.register_output(f"{prefix}/ctc-prior", prior)

    for lm_out_name, lm_name in _lms.items():
        lm = _get_lm_model(lm_name)

        for beam_size, prior_scale, lm_scale in [
            (12, 1.0, 1.0),
            # (1, 1.0, 1.0),
        ]:
            model = get_ctc_with_lm(
                ctc_model=ctc_model, prior=prior, prior_scale=prior_scale, language_model=lm, lm_scale=lm_scale
            )
            res = recog_model(
                task=task,
                model=model,
                recog_def=model_recog,
                config={
                    "beam_size": beam_size,
                    "recog_version": 4,
                    "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                },
                search_rqmt={"time": 24},
            )
            tk.register_output(
                f"{prefix}/recog-beam{beam_size}-lm_{lm_out_name}-lmScale{lm_scale}-priorScale{prior_scale}", res.output
            )


_sis_prefix: Optional[str] = None


def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

        prefix_name = get_setup_prefix_for_module(__name__)
    global _sis_prefix
    _sis_prefix = prefix_name


_called_ctc_py_once = False
_model_cache_by_name = {}


def _get_ctc_model(name: str) -> ModelWithCheckpoint:
    # noinspection PyProtectedMember
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import (
        py as ctc_py,
        _train_experiments as ctc_train_experiments,
    )

    global _called_ctc_py_once

    if name in _model_cache_by_name:
        return _model_cache_by_name[name]

    if not _called_ctc_py_once:
        from i6_experiments.users.zeyer.utils.sis_setup import disable_register_output

        with disable_register_output():
            ctc_py()
        _called_ctc_py_once = True

    exp = ctc_train_experiments[name]
    model = exp.get_last_fixed_epoch()
    _model_cache_by_name[name] = model
    return model


_lm_cache_by_name = {}
_called_lm_py_once = False


def _get_lm_model(name: str) -> ModelWithCheckpoint:
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.lm import py as lm_py
    from i6_experiments.users.zeyer.train_v3 import train_models_by_prefix

    global _called_lm_py_once

    if name in _lm_cache_by_name:
        return _lm_cache_by_name[name]

    if not _called_lm_py_once:
        from i6_experiments.users.zeyer.utils.sis_setup import disable_register_output

        with disable_register_output():
            lm_py()
        _called_lm_py_once = True

    exp = train_models_by_prefix["lm/" + name]
    model = exp.get_last_fixed_epoch()
    _lm_cache_by_name[name] = model
    return model


class ModelExt(Model):
    """
    Model extended with LM.

    Base model already can have a prior (static_prior).
    """

    def __init__(self, *, lm: Union[TransformerDecoder, Any], lm_scale: float, **kwargs):
        """
        :param lm: language model. We usually have TransformerDecoder, but any other type would also be ok
            when it has the same API.
        :param lm_scale: LM scale factor.
        :param kwargs: passed to super.
        """
        super().__init__(**kwargs)
        self.lm = lm
        self.lm_scale = lm_scale


def get_ctc_with_lm(
    *,
    ctc_model: ModelWithCheckpoint,
    prior: Optional[tk.Path] = None,
    prior_type: str = "prob",
    prior_scale: Optional[float] = None,
    language_model: ModelWithCheckpoint,
    lm_scale: float,
) -> ModelWithCheckpoint:
    # Keep CTC model config as-is, extend below for prior and LM.
    ctc_model_def = ctc_model.definition
    if isinstance(ctc_model_def, ModelDefWithCfg):
        config: Dict[str, Any] = ctc_model_def.config.copy()
    else:
        config = {}

    # Add prior.
    # Then the CTC Model log_probs_wb_from_logits will include the prior.
    if prior is not None:
        assert prior_scale is not None
    if prior_scale:
        assert prior is not None
        config.update(
            {
                "ctc_prior_type": "static",
                "ctc_prior_scale": prior_scale,
                "static_prior": {"type": prior_type, "file": prior},
            }
        )

    # Add LM.
    # LM has _model_def_dict in config. Put that as _lm_model_def_dict.
    config.update(
        {
            "_lm_model_def_dict": language_model.definition.config["_model_def_dict"],
            "lm_scale": lm_scale,
        }
    )
    config.setdefault("preload_from_files", {})["lm"] = {"prefix": "lm.", "filename": language_model.checkpoint}

    return ModelWithCheckpoint(
        definition=ModelDefWithCfg(model_def=ctc_model_ext_def, config=config),
        checkpoint=ctc_model.checkpoint,
    )


def ctc_model_ext_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    in_dim, epoch  # noqa
    config = get_global_config()  # noqa
    lm = rf.build_from_dict(config.typed_value("_lm_model_def_dict"), vocab_dim=target_dim)
    lm_scale = config.typed_value("lm_scale", None)
    assert isinstance(lm_scale, (int, float))
    return ModelExt(**_get_ctc_model_kwargs_from_global_config(target_dim=target_dim), lm=lm, lm_scale=lm_scale)


ctc_model_ext_def: ModelDef[Model]
ctc_model_ext_def.behavior_version = 21
ctc_model_ext_def.backend = "torch"
ctc_model_ext_def.batch_size_factor = _batch_size_factor


def model_recog(
    *,
    model: ModelExt,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    import tree
    from returnn.config import get_global_config

    config = get_global_config()
    beam_size = config.int("beam_size", 12)
    version = config.int("recog_version", 1)
    assert version == 4

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam

    # The label log probs include the AM and the (scaled) prior.
    label_log_prob = model.log_probs_wb_from_logits(logits)  # Batch, Spatial, VocabWB
    label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
    )
    label_log_prob_ta = TensorArray.unstack(label_log_prob, axis=enc_spatial_dim)  # t -> Batch, VocabWB

    lm_log_probs = rf.constant(0.0, dims=batch_dims_ + [model.target_dim])  # Batch, InBeam, Vocab
    lm_state = model.lm.default_initial_state(batch_dims=batch_dims_)  # Batch, InBeam, ...
    got_new_label = rf.constant(True, dims=batch_dims_, sparse_dim=model.wb_target_dim)  # Batch, InBeam -> 0|1
    target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)  # Batch, InBeam -> Vocab
    target_wb = rf.constant(
        model.blank_idx, dims=batch_dims_, sparse_dim=model.wb_target_dim
    )  # Batch, InBeam -> VocabWB

    max_seq_len = int(enc_spatial_dim.get_dim_value())
    seq_targets_wb = []
    seq_backrefs = []
    for t in range(max_seq_len):
        prev_target = target
        prev_target_wb = target_wb

        (prev_target_, lm_state_), packed_new_label_dim, packed_new_label_dim_map = _masked_select_tree(
            (prev_target, lm_state),
            mask=got_new_label,
            dims=batch_dims + [beam_dim],
        )

        lm_logits_, lm_state_ = model.lm(
            prev_target_,
            spatial_dim=single_step_dim,
            state=lm_state_,
        )  # Flat_Batch_InBeam, Vocab / ...
        lm_log_probs_ = rf.log_softmax(lm_logits_, axis=model.target_dim)  # Flat_Batch_InBeam, Vocab
        lm_log_probs_ *= model.lm_scale

        lm_log_probs, lm_state = _masked_scatter_tree(
            (lm_log_probs_, lm_state_),
            (lm_log_probs, lm_state),
            mask=got_new_label,
            dims=batch_dims + [beam_dim],
            in_dim=packed_new_label_dim,
            dim_map=packed_new_label_dim_map,
        )

        seq_log_prob = seq_log_prob + label_log_prob_ta[t]  # Batch, InBeam, VocabWB

        # Now add LM score. If prev align label (target_wb) is blank or != cur, add LM score, otherwise 0.
        seq_log_prob += rf.where(
            (prev_target_wb == model.blank_idx) | (prev_target_wb != rf.range_over_dim(model.wb_target_dim)),
            _target_dense_extend_blank(
                lm_log_probs,
                target_dim=model.target_dim,
                wb_target_dim=model.wb_target_dim,
                blank_idx=model.blank_idx,
                value=0.0,
            ),
            0.0,
        )  # Batch, InBeam -> VocabWB

        # Add LM EOS score in last frame.
        seq_log_prob += rf.where(
            t == enc_spatial_dim.get_dyn_size_ext_for_device(seq_log_prob.device) - 1,
            rf.gather(lm_log_probs, indices=model.eos_idx, axis=model.target_dim),
            0.0,
        )  # Batch, InBeam -> VocabWB

        seq_log_prob, (backrefs, target_wb), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{t}-beam"), axis=[beam_dim, model.wb_target_dim]
        )
        # seq_log_prob, backrefs, target_wb: Batch, Beam
        # backrefs -> InBeam.
        # target_wb -> VocabWB.
        seq_targets_wb.append(target_wb)
        seq_backrefs.append(backrefs)

        lm_state = tree.map_structure(functools.partial(_gather_backrefs, backrefs=backrefs), lm_state)
        prev_target = rf.gather(prev_target, indices=backrefs)  # Batch, Beam -> Vocab
        prev_target_wb = rf.gather(prev_target_wb, indices=backrefs)  # Batch, Beam -> VocabWB
        got_new_label = (target_wb != model.blank_idx) & (target_wb != prev_target_wb)  # Batch, Beam -> 0|1
        target = rf.where(
            got_new_label,
            _target_remove_blank(
                target_wb, target_dim=model.target_dim, wb_target_dim=model.wb_target_dim, blank_idx=model.blank_idx
            ),
            prev_target,
        )  # Batch, Beam -> Vocab

    # Backtrack via backrefs, resolve beams.
    seq_targets_wb_ = []
    indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target_wb in zip(seq_backrefs[::-1], seq_targets_wb[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_wb_.insert(0, rf.gather(target_wb, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets_wb__ = TensorArray(seq_targets_wb_[0])
    for target_wb in seq_targets_wb_:
        seq_targets_wb__ = seq_targets_wb__.push_back(target_wb)
    out_spatial_dim = enc_spatial_dim
    seq_targets_wb = seq_targets_wb__.stack(axis=out_spatial_dim)

    return seq_targets_wb, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog: RecogDef[Model]
model_recog.output_with_beam = True
model_recog.output_blank_label = "<blank>"
model_recog.batch_size_dependent = True  # our models currently just are batch-size-dependent...


def get_ctc_prior_probs(
    ctc_model: ModelWithCheckpoint, dataset: DatasetConfig, config: Optional[Dict[str, Any]] = None
) -> tk.Path:
    """
    :return: CTC prior, in prob space (not log prob)
    """
    # Note: there is also compute_model_softmax_prior_statistics,
    # which assumes a slightly different model API though,
    # and we must call log_probs_wb_from_logits to have it correct in any case.
    return collect_statistics(
        model=ctc_model, dataset=dataset, forward_def=_ctc_model_softmax_prior_returnn_forward, config=config
    ).mean


def _ctc_model_softmax_prior_returnn_forward(
    source: Tensor, /, in_spatial_dim: Dim, model: Model
) -> Tuple[Tensor, Dim]:
    """ForwardDef API"""
    import returnn.frontend as rf
    from returnn.tensor import Tensor, Dim

    logits, enc, enc_spatial_dim = model(source, in_spatial_dim=in_spatial_dim)
    assert isinstance(logits, Tensor) and isinstance(enc_spatial_dim, Dim)
    assert logits.feature_dim  # we expect a feature dim
    assert enc_spatial_dim in logits.dims
    log_probs = model.log_probs_wb_from_logits(logits)
    assert isinstance(log_probs, Tensor)
    probs = rf.exp(log_probs)  # the statistics take the average over this, thus prob space, not log prob
    return probs, enc_spatial_dim


T = TypeVar("T")


def _gather_backrefs(s: T, *, backrefs: Tensor) -> T:
    if isinstance(s, Tensor):
        if backrefs.sparse_dim in s.dims:
            return rf.gather(s, indices=backrefs)  # really the default case
        return s  # e.g. scalar or so, independent from beam
    if isinstance(s, Dim):
        if s.dimension:  # static
            return s
        assert backrefs.sparse_dim not in s.dyn_size_ext.dims  # currently not supported, also not expected
        return s
    raise TypeError(f"_gather_backrefs: unexpected type ({type(s)})")


def _masked_select_tree(s: T, *, mask: Tensor, dims: Sequence[Dim]) -> Tuple[T, Dim, Dict[Dim, Dim]]:
    import tree

    packed_new_label_dim = Dim(None, name="packed_new_label")  # Flat_Batch_InBeam
    packed_new_label_dim_map = {}
    tree.map_structure(
        functools.partial(
            _masked_select_prepare_dims,
            mask=mask,
            dims=dims,
            out_dim=packed_new_label_dim,
            dim_map=packed_new_label_dim_map,
        ),
        s,
    )
    s = tree.map_structure(
        functools.partial(
            _masked_select,
            mask=mask,
            dims=dims,
            out_dim=packed_new_label_dim,
            dim_map=packed_new_label_dim_map,
        ),
        s,
    )
    return s, packed_new_label_dim, packed_new_label_dim_map


def _masked_select_prepare_dims(s, *, mask: Tensor, dims: Sequence[Dim], out_dim: Dim, dim_map: Dict[Dim, Dim]):
    if isinstance(s, Tensor):
        return s  # ignored at this stage
    if isinstance(s, Dim):
        if s.dimension:  # static
            return s
        if not any(d in s.dyn_size_ext.dims for d in dims):
            return s
        if s in dim_map:
            return dim_map[s]
        new_dyn_size = _masked_select(s.dyn_size_ext, mask=mask, dims=dims, out_dim=out_dim, dim_map=dim_map)
        new_dim = Dim(new_dyn_size, name=s.name + "_")
        dim_map[s] = new_dim
        return new_dim
    raise TypeError(f"_masked_select_prepare_dims: unexpected type ({type(s)})")


def _masked_select(s: T, *, mask: Tensor, dims: Sequence[Dim], out_dim: Dim, dim_map: Dict[Dim, Dim]) -> T:
    if isinstance(s, Tensor):
        if not any(d in s.dims for d in dims):
            return s  # e.g. scalar or so, independent from dims
        # For the masked_select, we need that all masked dims are present, so add them if not.
        # (E.g., when we mask [batch,beam], but we only have [batch], we need to add the beam dim.)
        if any(d not in s.dims for d in dims):
            s = rf.expand_dims(s, dims=[d for d in dims if d not in s.dims])
        # The packing itself (masked_select).
        s, _ = rf.masked_select(s, mask=mask, dims=dims, out_dim=out_dim)
        # In the resulting tensor, potentially replace dims.
        # In addition to the dim replacement, we also might need to slice, as the size might be smaller.
        if any(d in dim_map for d in s.dims):
            for d in s.dims:
                if d in dim_map:
                    s = rf.slice(s, axis=d, size=dim_map[d])
        return s
    if isinstance(s, Dim):
        if s.dimension:  # static
            return s
        if not any(d in s.dyn_size_ext.dims for d in dims):
            return s
        assert s in dim_map
        return dim_map[s]
    raise TypeError(f"_gather_backrefs: unexpected type ({type(s)})")


def _masked_scatter_tree(
    s: T, backup: T, *, mask: Tensor, dims: Sequence[Dim], in_dim: Dim, dim_map: Dict[Dim, Dim]
) -> T:
    import tree

    reverse_dict_map = {v: k for k, v in dim_map.items()}

    s = tree.map_structure(
        functools.partial(_masked_scatter, mask=mask, dims=dims, in_dim=in_dim, reverse_dict_map=reverse_dict_map),
        s,
        backup,
    )
    return s


def _masked_scatter(
    s: T, backup: T, *, mask: Tensor, dims: Sequence[Dim], in_dim: Dim, reverse_dim_map: Dict[Dim, Dim]
) -> T:
    if isinstance(s, Tensor):
        if in_dim not in s.dims:
            return s  # e.g. scalar or so, independent from masking
        # Do the reverse of _masked_select above.
        # First replace the dims back.
        if any(d in reverse_dim_map for d in s.dims):
            for d in s.dims:
                if d in reverse_dim_map:
                    s = _expand_slice(s, axis=d, expanded_size=reverse_dim_map[d])
        # The unpacking itself (reversing the masked_select, i.e. masked_scatter).
        assert isinstance(backup, Tensor)
        s = rf.masked_scatter(s, backup, mask=mask, dims=dims, in_dim=in_dim)
        # Now remove potential added dims.
        for d in s.dims:
            if d not in backup.dims:
                s = rf.gather(s, axis=d, indices=0)
        return s
    if isinstance(s, Dim):
        if s.dimension:  # static
            return s
        if not any(d in s.dyn_size_ext.dims for d in dims):
            return s
        assert s in reverse_dim_map
        return reverse_dim_map[s]
    raise TypeError(f"_masked_scatter: unexpected type ({type(s)})")


def _expand_slice(source: Tensor, axis: Dim, expanded_size: Dim) -> Tensor:
    res, _ = rf.pad(
        source,
        axes=[axis],
        padding=[(0, expanded_size.get_dim_value_tensor() - axis.get_dim_value_tensor())],
        out_dims=[expanded_size],
    )
    return res


def _target_remove_blank(target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int) -> Tensor:
    assert target.sparse_dim == wb_target_dim
    assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
    return rf.set_sparse_dim(target, target_dim)


def _target_dense_extend_blank(
    target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int, value: float
) -> Tensor:
    assert target_dim in target.dims
    assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
    res, _ = rf.pad(target, axes=[target_dim], padding=[(0, 1)], out_dims=[wb_target_dim], value=value)
    return res
