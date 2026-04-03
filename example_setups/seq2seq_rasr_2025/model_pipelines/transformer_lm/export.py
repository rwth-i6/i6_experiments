__all__ = [
    "export_model_stateless",
    "export_model_kv_cached",
    "export_scorer",
    "export_state_initializer",
    "export_state_updater",
]

import torch
from i6_core.returnn import CodeWrapper, PtCheckpoint
from returnn.tensor.tensor_dict import TensorDict
from sisyphus import tk

from i6_experiments.common.setups.serialization import Import

from ..common.onnx_export import export_model as _export_model
from ..common.pytorch_modules import lengths_to_padding_mask
from ..common.serializers import get_model_serializers
from .pytorch_modules import (
    TransformerLm,
    TransformerLmConfig,
    TransformerLmOnnxWrapper,
    TransformerLmScorer,
    TransformerLmStateInitializer,
    TransformerLmStateUpdater,
)


def _forward_step(*, model: TransformerLm, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    tokens = extern_data["tokens"].raw_tensor
    assert isinstance(tokens, torch.Tensor)
    assert extern_data["tokens"].dims[1].dyn_size_ext is not None
    tokens_size = extern_data["tokens"].dims[1].dyn_size_ext.raw_tensor  # [B]
    assert tokens_size is not None

    seq_mask = lengths_to_padding_mask(tokens_size)

    logits = model.forward(input=tokens, seq_mask=seq_mask)  # [B, N, V]

    batch_size = logits.size(0)
    last_logits = logits[torch.arange(batch_size), tokens_size.long() - 1]
    scores = -torch.nn.functional.log_softmax(last_logits, dim=-1)

    run_ctx.mark_as_output(name="scores", tensor=scores)


def _forward_step_v2(*, model: TransformerLmOnnxWrapper, extern_data: TensorDict, **_):
    import returnn.frontend as rf
    from returnn.tensor.dim import batch_dim

    run_ctx = rf.get_run_ctx()

    tokens = extern_data["tokens"].raw_tensor
    assert isinstance(tokens, torch.Tensor)
    assert extern_data["tokens"].dims[1].dyn_size_ext is not None
    tokens_size = extern_data["tokens"].dims[1].dyn_size_ext.raw_tensor  # [B]
    assert tokens_size is not None
    tokens_size = tokens_size.long()

    assert extern_data["state_l000_k_in"].dims[1].dyn_size_ext is not None
    prefix_lens = extern_data["state_l000_k_in"].dims[1].dyn_size_ext.raw_tensor  # [B]
    assert prefix_lens is not None
    prefix_lens = prefix_lens.long()

    kv_cache = []
    for layer in range(model.num_layers):
        kv_cache.append(extern_data[f"state_l{layer:03d}_k_in"].raw_tensor)
        kv_cache.append(extern_data[f"state_l{layer:03d}_v_in"].raw_tensor)

    scores, *new_kv_cache = model.forward(tokens, tokens_size, prefix_lens, *kv_cache)

    run_ctx.mark_as_output(name="scores", tensor=scores)

    suffix_time_dim = rf.Tensor("suffix_time", dims=[batch_dim], raw_tensor=tokens_size.long(), dtype="int32")

    idx = 0
    for layer in range(model.num_layers):
        run_ctx.mark_as_output(name=f"state_l{layer:03d}_k_out", tensor=new_kv_cache[idx])
        run_ctx.mark_as_output(name=f"state_l{layer:03d}_v_out", tensor=new_kv_cache[idx + 1])
        if run_ctx.expected_outputs is not None:
            run_ctx.expected_outputs[f"state_l{layer:03d}_k_out"].dims[1].dyn_size_ext = suffix_time_dim
            run_ctx.expected_outputs[f"state_l{layer:03d}_v_out"].dims[1].dyn_size_ext = suffix_time_dim
        idx += 2


def _scorer_forward_step(*, model: TransformerLmScorer, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    transformer_out = extern_data["transformer_out"].raw_tensor
    assert transformer_out is not None

    scores = model.forward(transformer_out=transformer_out)

    run_ctx.mark_as_output(name="scores", tensor=scores)


def _state_initializer_forward_step(*, model: TransformerLmStateInitializer, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    transformer_out, *kv_cache = model.forward()

    state_length = torch.full(
        (kv_cache[0].size(0),), kv_cache[1].size(1), dtype=torch.int32, device=transformer_out.device
    )

    run_ctx.mark_as_output(name="transformer_out", tensor=transformer_out)
    run_ctx.mark_as_output(name="state_length", tensor=state_length)

    idx = 0
    for layer in range(model.num_layers):
        if run_ctx.expected_outputs is not None:
            assert run_ctx.expected_outputs[f"state_l{layer:03d}_k"].dims[1].dyn_size_ext is not None
            run_ctx.expected_outputs[f"state_l{layer:03d}_k"].dims[1].dyn_size_ext.raw_tensor = state_length
            assert run_ctx.expected_outputs[f"state_l{layer:03d}_v"].dims[1].dyn_size_ext is not None
            run_ctx.expected_outputs[f"state_l{layer:03d}_v"].dims[1].dyn_size_ext.raw_tensor = state_length
        run_ctx.mark_as_output(name=f"state_l{layer:03d}_k", tensor=kv_cache[idx])
        run_ctx.mark_as_output(name=f"state_l{layer:03d}_v", tensor=kv_cache[idx + 1])
        idx += 2


def _state_updater_forward_step(*, model: TransformerLmStateUpdater, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    token = extern_data["token"].raw_tensor
    assert token is not None

    assert extern_data["state_l000_k_in"].dims[1].dyn_size_ext is not None
    state_length = extern_data["state_l000_k_in"].dims[1].dyn_size_ext.raw_tensor
    assert state_length is not None

    kv_cache = []
    for layer in range(model.num_layers):
        state_k_in = extern_data[f"state_l{layer:03d}_k_in"].raw_tensor
        assert state_k_in is not None
        state_v_in = extern_data[f"state_l{layer:03d}_v_in"].raw_tensor
        assert state_v_in is not None
        kv_cache.extend([state_k_in, state_v_in])

    transformer_out, *new_kv_cache = model.forward(token, state_length.long(), *kv_cache)

    new_lengths = state_length + 1

    run_ctx.mark_as_output(name="transformer_out", tensor=transformer_out)
    run_ctx.mark_as_output(name="state_length_out", tensor=new_lengths)

    idx = 0
    for layer in range(model.num_layers):
        if run_ctx.expected_outputs is not None:
            assert run_ctx.expected_outputs[f"state_l{layer:03d}_k_out"].dims[1].dyn_size_ext is not None
            run_ctx.expected_outputs[f"state_l{layer:03d}_k_out"].dims[1].dyn_size_ext.raw_tensor = new_lengths
            assert run_ctx.expected_outputs[f"state_l{layer:03d}_v_out"].dims[1].dyn_size_ext is not None
            run_ctx.expected_outputs[f"state_l{layer:03d}_v_out"].dims[1].dyn_size_ext.raw_tensor = new_lengths
        run_ctx.mark_as_output(name=f"state_l{layer:03d}_k_out", tensor=new_kv_cache[idx])
        run_ctx.mark_as_output(name=f"state_l{layer:03d}_v_out", tensor=new_kv_cache[idx + 1])
        idx += 2


def export_model_stateless(model_config: TransformerLmConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=TransformerLm, model_config=model_config)

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(f"{_forward_step.__module__}.{_forward_step.__name__}", import_as="forward_step"),
        checkpoint=checkpoint,
        returnn_config_dict={
            "extern_data": {
                "tokens": {
                    "dim": model_config.vocab_dim,
                    "sparse": True,
                    "dtype": "int32",
                },
            },
            "model_outputs": {
                "scores": {
                    "dim": model_config.vocab_dim,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
            },
        },
        input_names=["tokens", "tokens:size1"],
        output_names=["scores"],
    )


def export_model_kv_cached(model_config: TransformerLmConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=TransformerLmOnnxWrapper, model_config=model_config)

    extern_data = {}
    input_names = []
    model_outputs = {
        "scores": {
            "dim_tags": CodeWrapper("(batch_dim, dim_suffix_time, dim_score_feature)"),
            "dtype": "float32",
        }
    }
    output_names = ["scores", "scores:size1"]

    metadata = {}

    for layer in range(model_config.num_layers):
        extern_data[f"state_l{layer:03d}_k_in"] = {
            "dim_tags": CodeWrapper("(batch_dim, dim_prefix_time, dim_state_feature)"),
            "dtype": "float32",
        }
        extern_data[f"state_l{layer:03d}_v_in"] = {
            "dim_tags": CodeWrapper("(batch_dim, dim_prefix_time, dim_state_feature)"),
            "dtype": "float32",
        }
        input_names.append(f"state_l{layer:03d}_k_in")
        if layer == 0:
            input_names.append(f"state_l{layer:03d}_k_in:size1")
        input_names.append(f"state_l{layer:03d}_v_in")

        model_outputs[f"state_l{layer:03d}_k_out"] = {
            "dim_tags": CodeWrapper("(batch_dim, dim_suffix_time, dim_state_feature)"),
            "dtype": "float32",
        }
        model_outputs[f"state_l{layer:03d}_v_out"] = {
            "dim_tags": CodeWrapper("(batch_dim, dim_suffix_time, dim_state_feature)"),
            "dtype": "float32",
        }
        output_names += [
            f"state_l{layer:03d}_k_out",
            f"state_l{layer:03d}_k_out:size1",
            f"state_l{layer:03d}_v_out",
            f"state_l{layer:03d}_v_out:size1",
        ]

        metadata[f"STATE_state_l{layer:03d}_k_in"] = f"state_l{layer:03d}_k_out"
        metadata[f"STATE_state_l{layer:03d}_v_in"] = f"state_l{layer:03d}_v_out"

    extern_data = {
        "tokens": {
            "dim_tags": CodeWrapper("(batch_dim, dim_suffix_time)"),
            "dim": model_config.vocab_dim,
            "sparse": True,
            "dtype": "int32",
        },
        **extern_data,
    }
    input_names = ["tokens", "tokens:size1"] + input_names

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_forward_step_v2.__module__}.{_forward_step_v2.__name__}", import_as="forward_step"
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "dim_suffix_time": CodeWrapper('Dim(name="suffix_time", dimension=None)'),
            "dim_score_feature": CodeWrapper(f'Dim(name="suffix_time", dimension={model_config.vocab_dim})'),
            "dim_prefix_time": CodeWrapper('Dim(name="prefix_time", dimension=None)'),
            "dim_state_feature": CodeWrapper(f'Dim(name="state_feature", dimension={model_config.hid_dim})'),
            "extern_data": extern_data,
            "model_outputs": model_outputs,
        },
        input_names=input_names,
        output_names=output_names,
        metadata=metadata,
    )


def export_scorer(model_config: TransformerLmConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=TransformerLmScorer, model_config=model_config)

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_scorer_forward_step.__module__}.{_scorer_forward_step.__name__}", import_as="forward_step"
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "extern_data": {
                "transformer_out": {
                    "dim": model_config.hid_dim,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
            },
            "model_outputs": {
                "scores": {
                    "dim": model_config.vocab_dim,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
            },
        },
        input_names=["transformer_out"],
        output_names=["scores"],
        metadata={"transformer_out": "transformer_out"},
    )


def export_state_initializer(model_config: TransformerLmConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=TransformerLmStateInitializer, model_config=model_config)

    model_outputs = {
        "transformer_out": {
            "dim_tags": CodeWrapper("(batch_dim, dim_state_feature)"),
            "dtype": "float32",
        },
        "state_length": CodeWrapper("state_length_tensor"),
    }
    output_names = ["transformer_out", "state_length"]
    metadata = {
        "transformer_out": "transformer_out",
        "state_length": "state_length",
    }

    for layer in range(model_config.num_layers):
        model_outputs[f"state_l{layer:03d}_k"] = {
            "dim_tags": CodeWrapper("(batch_dim, dim_state_length, dim_state_feature)"),
            "dtype": "float32",
        }
        model_outputs[f"state_l{layer:03d}_v"] = {
            "dim_tags": CodeWrapper("(batch_dim, dim_state_length, dim_state_feature)"),
            "dtype": "float32",
        }
        output_names += [
            f"state_l{layer:03d}_k",
            f"state_l{layer:03d}_k:size1",
            f"state_l{layer:03d}_v",
            f"state_l{layer:03d}_v:size1",
        ]
        metadata[f"state_l{layer:03d}_k"] = f"state_l{layer:03d}_k"
        metadata[f"state_l{layer:03d}_v"] = f"state_l{layer:03d}_v"

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_state_initializer_forward_step.__module__}.{_state_initializer_forward_step.__name__}",
            import_as="forward_step",
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "state_length_tensor": CodeWrapper("Tensor(name='state_length', dim_tags=[batch_dim], dtype='int32')"),
            "dim_state_length": CodeWrapper(
                'Dim(name="state_length", dimension=None, dyn_size_ext=state_length_tensor)'
            ),
            "dim_state_feature": CodeWrapper(f'Dim(name="state_feature", dimension={model_config.hid_dim})'),
            "extern_data": {},
            "model_outputs": model_outputs,
        },
        extra_imports=[Import("returnn.tensor.Tensor")],
        input_names=[],
        output_names=output_names,
        metadata=metadata,
    )


def export_state_updater(model_config: TransformerLmConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=TransformerLmStateUpdater, model_config=model_config)

    extern_data = {
        "token": {
            "dim": model_config.vocab_dim,
            "time_dim_axis": None,
            "sparse": True,
            "dtype": "int32",
        },
    }
    input_names = ["token"]
    model_outputs = {
        "transformer_out": {
            "dim": model_config.hid_dim,
            "time_dim_axis": None,
            "dtype": "float32",
        },
        "state_length_out": CodeWrapper("state_length_out_tensor"),
    }
    output_names = ["transformer_out", "state_length_out"]
    metadata = {
        "transformer_out": "transformer_out",
        "state_length_out": "state_length",
    }

    for layer in range(model_config.num_layers):
        extern_data[f"state_l{layer:03d}_k_in"] = {
            "dim_tags": CodeWrapper("(batch_dim, dim_state_length_in, dim_state_feature)"),
            "dtype": "float32",
        }
        extern_data[f"state_l{layer:03d}_v_in"] = {
            "dim_tags": CodeWrapper("(batch_dim, dim_state_length_in, dim_state_feature)"),
            "dtype": "float32",
        }
        input_names.append(f"state_l{layer:03d}_k_in")
        if layer == 0:
            input_names.append(f"state_l{layer:03d}_k_in:size1")
        input_names.append(f"state_l{layer:03d}_v_in")

        model_outputs[f"state_l{layer:03d}_k_out"] = {
            "dim_tags": CodeWrapper("(batch_dim, dim_state_length_out, dim_state_feature)"),
            "dtype": "float32",
        }
        model_outputs[f"state_l{layer:03d}_v_out"] = {
            "dim_tags": CodeWrapper("(batch_dim, dim_state_length_out, dim_state_feature)"),
            "dtype": "float32",
        }
        output_names += [
            f"state_l{layer:03d}_k_out",
            f"state_l{layer:03d}_k_out:size1",
            f"state_l{layer:03d}_v_out",
            f"state_l{layer:03d}_v_out:size1",
        ]
        metadata[f"state_l{layer:03d}_k_in"] = f"state_l{layer:03d}_k"
        metadata[f"state_l{layer:03d}_v_in"] = f"state_l{layer:03d}_v"
        metadata[f"state_l{layer:03d}_k_out"] = f"state_l{layer:03d}_k"
        metadata[f"state_l{layer:03d}_v_out"] = f"state_l{layer:03d}_v"
        if layer == 0:
            metadata[f"state_l{layer:03d}_k_in:size1"] = "state_length"

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_state_updater_forward_step.__module__}.{_state_updater_forward_step.__name__}",
            import_as="forward_step",
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "state_length_out_tensor": CodeWrapper(
                "Tensor(name='state_length_out', dim_tags=[batch_dim], dtype='int32')"
            ),
            "dim_state_length_in": CodeWrapper('Dim(name="state_length_in", dimension=None)'),
            "dim_state_length_out": CodeWrapper(
                'Dim(name="state_length_out", dimension=None, dyn_size_ext=state_length_out_tensor)'
            ),
            "dim_state_feature": CodeWrapper(f'Dim(name="state_feature", dimension={model_config.hid_dim})'),
            "extern_data": extern_data,
            "model_outputs": model_outputs,
        },
        extra_imports=[Import("returnn.tensor.Tensor")],
        input_names=input_names,
        output_names=output_names,
        metadata=metadata,
    )
