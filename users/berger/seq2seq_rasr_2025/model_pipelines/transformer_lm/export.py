__all__ = [
    "export_model_stateless",
    "export_model_kv_cached",
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
)


def export_model_stateless(model_config: TransformerLmConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=TransformerLm, model_config=model_config)

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_forward_step_stateless.__module__}.{_forward_step_stateless.__name__}", import_as="forward_step"
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "tensor_tokens_length": CodeWrapper('Tensor(name="tokens_length", dims=[batch_dim], dtype="int32")'),
            "dim_tokens_length": CodeWrapper('Dim(dimension=tensor_tokens_length, name="tokens_length")'),
            "dim_vocab": CodeWrapper(f'Dim(dimension={model_config.vocab_dim}, name="vocab")'),
            "extern_data": {
                "tokens": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_tokens_length)"),
                    "sparse_dim": CodeWrapper("dim_vocab"),
                    "dtype": "int32",
                },
            },
            "model_outputs": {
                "scores": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_vocab)"),
                    "dtype": "float32",
                },
            },
        },
        input_names=["tokens", "tokens:size1"],
        output_names=["scores"],
    )


def export_model_kv_cached(model_config: TransformerLmConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=TransformerLmOnnxWrapper, model_config=model_config)

    extern_data = {
        "tokens": {
            "dim_tags": CodeWrapper("(batch_dim, dim_suffix_time)"),
            "sparse_dim": CodeWrapper("dim_vocab"),
            "dtype": "int32",
        },
    }
    input_names = ["tokens", "tokens:size1"]
    model_outputs = {
        "scores": {
            "dim_tags": CodeWrapper("(batch_dim, dim_suffix_time, dim_vocab)"),
            "dtype": "float32",
        }
    }
    output_names = ["scores", "scores:size1"]

    metadata = {}

    for layer in range(model_config.num_layers):
        extern_data[f"state_l{layer:03d}_k_in"] = {
            "dim_tags": CodeWrapper("(batch_dim, dim_prefix_time, dim_hidden)"),
            "dtype": "float32",
        }
        extern_data[f"state_l{layer:03d}_v_in"] = {
            "dim_tags": CodeWrapper("(batch_dim, dim_prefix_time, dim_hidden)"),
            "dtype": "float32",
        }
        input_names.append(f"state_l{layer:03d}_k_in")
        if layer == 0:
            input_names.append(f"state_l{layer:03d}_k_in:size1")
        input_names.append(f"state_l{layer:03d}_v_in")

        model_outputs[f"state_l{layer:03d}_k_out"] = {
            "dim_tags": CodeWrapper("(batch_dim, dim_suffix_time, dim_hidden)"),
            "dtype": "float32",
        }
        model_outputs[f"state_l{layer:03d}_v_out"] = {
            "dim_tags": CodeWrapper("(batch_dim, dim_suffix_time, dim_hidden)"),
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

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_forward_step_kv_cache.__module__}.{_forward_step_kv_cache.__name__}", import_as="forward_step"
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "tensor_prefix_time": CodeWrapper('Tensor(name="prefix_time", dims=[batch_dim], dtype="int32")'),
            "dim_prefix_time": CodeWrapper('Dim(dimension=tensor_prefix_time, name="prefix_time")'),
            "tensor_suffix_time": CodeWrapper('Tensor(name="suffix_time", dims=[batch_dim], dtype="int32")'),
            "dim_suffix_time": CodeWrapper('Dim(dimension=tensor_suffix_time, name="suffix_time")'),
            "dim_vocab": CodeWrapper(f'Dim(dimension={model_config.vocab_dim}, name="vocab")'),
            "dim_hidden": CodeWrapper(f'Dim(name="hidden", dimension={model_config.hid_dim})'),
            "extern_data": extern_data,
            "model_outputs": model_outputs,
        },
        input_names=input_names,
        output_names=output_names,
        metadata=metadata,
    )


def _forward_step_stateless(*, model: TransformerLm, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    tokens = extern_data["tokens"].raw_tensor
    assert tokens is not None

    assert extern_data["tokens"].dims[1].dyn_size_ext is not None
    tokens_size = extern_data["tokens"].dims[1].dyn_size_ext.raw_tensor  # [B]
    assert tokens_size is not None

    seq_mask = lengths_to_padding_mask(tokens_size)

    logits = model(input=tokens, seq_mask=seq_mask)  # [B, N, V]

    batch_size = logits.size(0)
    last_logits = logits[torch.arange(batch_size), tokens_size.long() - 1]
    scores = -torch.nn.functional.log_softmax(last_logits, dim=-1)

    run_ctx.mark_as_output(name="scores", tensor=scores)


def _forward_step_kv_cache(*, model: TransformerLmOnnxWrapper, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    tokens = extern_data["tokens"].raw_tensor
    assert tokens is not None

    assert extern_data["tokens"].dims[1].dyn_size_ext is not None
    tokens_size = extern_data["tokens"].dims[1].dyn_size_ext.raw_tensor  # [B]
    assert tokens_size is not None

    assert extern_data["state_l000_k_in"].dims[1].dyn_size_ext is not None
    prefix_lens = extern_data["state_l000_k_in"].dims[1].dyn_size_ext.raw_tensor  # [B]
    assert prefix_lens is not None

    kv_cache = []
    for layer in range(model.num_layers):
        kv_cache.append(extern_data[f"state_l{layer:03d}_k_in"].raw_tensor)
        kv_cache.append(extern_data[f"state_l{layer:03d}_v_in"].raw_tensor)

    scores, *new_kv_cache = model(tokens, tokens_size, prefix_lens.long(), *kv_cache)

    run_ctx.mark_as_output(name="scores", tensor=scores)

    idx = 0
    for layer in range(model.num_layers):
        run_ctx.mark_as_output(name=f"state_l{layer:03d}_k_out", tensor=new_kv_cache[idx])
        run_ctx.mark_as_output(name=f"state_l{layer:03d}_v_out", tensor=new_kv_cache[idx + 1])
        idx += 2
