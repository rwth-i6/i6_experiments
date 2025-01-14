"""
CTC recognition with LM
"""

from typing import Optional, Any, Tuple, Dict

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf

from sisyphus import tk

from returnn_common.datasets_old_2022_10.interface import DatasetConfig
from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, RecogDef, ModelWithCheckpoint

from i6_experiments.users.zeyer.recog import recog_model
from i6_experiments.users.zeyer.collect_model_dataset_stats import collect_statistics

from .ctc import Model, ctc_model_def, _batch_size_factor


_ctc_model_name = (
    # last epoch: {"dev-clean": 2.38, "dev-other": 5.67, "test-clean": 2.63, "test-other": 5.93}
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

        # Our own beam search implementation.
        for beam_size, prior_scale, lm_scale in [
            # (12, 1.0, 1.0),
            (12, 0.0, 1.0),
            # (1, 0.0, 1.0),
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
                    "recog_version": 6,
                    "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                },
                search_rqmt={"time": 24},
            )
            tk.register_output(
                f"{prefix}/recog-beam{beam_size}-lm_{lm_out_name}-lmScale{lm_scale}-priorScale{prior_scale}", res.output
            )

        # Flashlight beam search implementation.
        # Play around with beam size here.
        for prior_scale, lm_scale in [
            (0.0, 1.0),
            # (0.2, 2.0),
        ]:
            model = get_ctc_with_lm(
                ctc_model=ctc_model, prior=prior, prior_scale=prior_scale, language_model=lm, lm_scale=lm_scale
            )
            for name, opts in [
                # This takes forever (more than 2h for only the first (longest) seq of the corpora),
                # and then at some point runs out of CPU memory (OOM killer kills it).
                # (
                #     "beam1024-beamToken128-cache1024",
                #     {
                #         "n_best": 32,
                #         "beam_size": 1024,
                #         "beam_size_token": 128,
                #         "beam_threshold": 14,
                #         "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                #         "torch_amp": {"dtype": "bfloat16"},
                #         "lm_state_lru_initial_cache_size": 1024,
                #     },
                # ),
                (  # {"dev-clean": 3.51, "dev-other": 5.79, "test-clean": 3.66, "test-other": 6.27}
                    "beam16-beamToken16-cache1024",
                    {
                        "n_best": 16,
                        "beam_size": 16,
                        "beam_size_token": 16,
                        "beam_threshold": 14,
                        "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                        "torch_amp": {"dtype": "bfloat16"},
                        "lm_state_lru_initial_cache_size": 1024,  # -- default
                    },
                ),
                (
                    "beam16-beamToken16-cache2pow16",
                    {
                        "n_best": 16,
                        "beam_size": 16,
                        "beam_size_token": 16,
                        "beam_threshold": 14,
                        "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                        "torch_amp": {"dtype": "bfloat16"},
                        "lm_state_lru_initial_cache_size": 2**16,
                    },
                ),
                (
                    "beam16-beamToken128-cache1024",
                    {
                        "n_best": 16,
                        "beam_size": 16,
                        "beam_size_token": 128,
                        "beam_threshold": 14,
                        "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                        "torch_amp": {"dtype": "bfloat16"},
                        "lm_state_lru_initial_cache_size": 1024,  # -- default
                    },
                ),
                (
                    "beam4-beamToken16-cache1024",
                    {
                        "n_best": 16,
                        "beam_size": 4,
                        "beam_size_token": 16,
                        "beam_threshold": 14,
                        "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                        "torch_amp": {"dtype": "bfloat16"},
                        "lm_state_lru_initial_cache_size": 1024,
                    },
                ),
                (
                    "beam4-beamToken16-cache1024-f32",
                    {
                        "n_best": 16,
                        "beam_size": 4,
                        "beam_size_token": 16,
                        "beam_threshold": 14,
                        "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                        "lm_state_lru_initial_cache_size": 1024,
                    },
                ),
            ]:
                res = recog_model(
                    task=task,
                    model=model,
                    recog_def=model_recog_flashlight,
                    config=opts,
                    search_rqmt={"cpu": 4, "mem": 30, "time": 24, "gpu_mem": 24},
                )
                tk.register_output(
                    f"{prefix}/recog-fl-{name}-lm_{lm_out_name}-lmScale{lm_scale}-priorScale{prior_scale}",
                    res.output,
                )

        # Flashlight beam search implementation.
        # Play around with scales.
        for prior_scale, lm_scale in [
            (0.0, 0.0),
            (0.0, 1.0),
            (0.2, 2.0),
            (0.2, 1.0),
            (0.5, 1.0),
            (0.7, 2.0),
        ]:
            model = get_ctc_with_lm(
                ctc_model=ctc_model, prior=prior, prior_scale=prior_scale, language_model=lm, lm_scale=lm_scale
            )
            res = recog_model(
                task=task,
                model=model,
                recog_def=model_recog_flashlight,
                config={
                    "n_best": 16,
                    "beam_size": 16,
                    "beam_size_token": 128,
                    "beam_threshold": 14,
                    "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                    "torch_amp": {"dtype": "bfloat16"},
                },
                search_rqmt={"cpu": 4, "mem": 30, "time": 24, "gpu_mem": 24},
            )
            tk.register_output(
                f"{prefix}/recog-fl-beamToken128-lm_{lm_out_name}-lmScale{lm_scale}-priorScale{prior_scale}", res.output
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
    model = ctc_model_def(epoch=epoch, in_dim=in_dim, target_dim=target_dim)
    model.lm = lm
    model.lm_scale = lm_scale
    return model


ctc_model_ext_def: ModelDef[Model]
ctc_model_ext_def.behavior_version = 21
ctc_model_ext_def.backend = "torch"
ctc_model_ext_def.batch_size_factor = _batch_size_factor


# Just an alias, but keep func here to not break hashes.
def model_recog(
    *,
    model: Model,
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
    from .recog_ext.ctc import model_recog

    return model_recog(model=model, data=data, data_spatial_dim=data_spatial_dim)


# RecogDef API
model_recog: RecogDef[Model]
model_recog.output_with_beam = True
model_recog.output_blank_label = "<blank>"
model_recog.batch_size_dependent = True  # our models currently just are batch-size-dependent...


# Just an alias, but keep func here to not break hashes.
def model_recog_flashlight(
    *,
    model: Model,
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
    from .recog_ext.flashlight_neural_lm import model_recog_flashlight

    return model_recog_flashlight(model=model, data=data, data_spatial_dim=data_spatial_dim)


# RecogDef API
model_recog_flashlight: RecogDef[Model]
model_recog_flashlight.output_with_beam = True
model_recog_flashlight.output_blank_label = "<blank>"
model_recog_flashlight.batch_size_dependent = True  # our models currently just are batch-size-dependent...


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
