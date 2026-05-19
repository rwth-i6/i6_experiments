__all__ = ["export_initializer_model", "export_step_model"]

from i6_core.returnn import CodeWrapper, PtCheckpoint
from returnn.tensor.tensor_dict import TensorDict
from sisyphus import tk

from i6_experiments.common.setups.returnn_pytorch.serialization import PyTorchModel
from i6_experiments.common.setups.serialization import Collection, Import

from ..common.onnx_export import export_model as _export_model
from .pytorch_modules import SpeechLmCtc, SpeechLmInitializer, SpeechLmStep


def export_initializer_model(model_kwargs: dict, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = Collection(
        [
            Import(f"{SpeechLmInitializer.__module__}.{SpeechLmInitializer.__name__}"),
            PyTorchModel(
                model_class_name=SpeechLmInitializer.__name__,
                model_kwargs=model_kwargs,
            ),
        ]
    )

    extern_data = {
        "encoder_states": {
            "dim_tags": CodeWrapper("(batch_dim, dim_encoder_time, dim_encoder_feature)"),
            "dtype": "float32",
        },
        "initial_prompt": {
            "dim_tags": CodeWrapper("(batch_dim, dim_initial_time)"),
            "sparse_dim": CodeWrapper("dim_vocab"),
            "dtype": "int32",
        },
        "suffix_prompt": {
            "dim_tags": CodeWrapper("(batch_dim, dim_suffix_time)"),
            "sparse_dim": CodeWrapper("dim_vocab"),
            "dtype": "int32",
        },
    }
    input_names = [
        "encoder_states",
        "encoder_states:size1",
        "initial_prompt",
        "initial_prompt:size1",
        "suffix_prompt",
        "suffix_prompt:size1",
    ]
    model_outputs = {
        "scores": {
            "dim_tags": CodeWrapper("(batch_dim, dim_vocab)"),
            "dtype": "float32",
        }
    }
    output_names = ["scores"]

    metadata = {}

    for layer in range(24):  # TODO: set dynamically
        model_outputs[f"state_l{layer:03d}_k_out"] = {
            "dim_tags": CodeWrapper("(batch_dim, dim_num_heads, dim_time, dim_head)"),
            "dtype": "float32",
        }
        model_outputs[f"state_l{layer:03d}_v_out"] = {
            "dim_tags": CodeWrapper("(batch_dim, dim_num_heads, dim_time, dim_head)"),
            "dtype": "float32",
        }
        output_names += [
            f"state_l{layer:03d}_k_out",
            f"state_l{layer:03d}_k_out:size2",
            f"state_l{layer:03d}_v_out",
            f"state_l{layer:03d}_v_out:size2",
        ]

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_forward_step_initializer.__module__}.{_forward_step_initializer.__name__}", import_as="forward_step"
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "dim_vocab": CodeWrapper('Dim(dimension=151936, name="vocab")'),  # TODO: set dynamically
            "tensor_initial_time": CodeWrapper('Tensor(name="initial_time", dims=[batch_dim], dtype="int32")'),
            "dim_initial_time": CodeWrapper('Dim(dimension=tensor_initial_time, name="initial_time")'),
            "tensor_encoder_time": CodeWrapper('Tensor(name="encoder_time", dims=[batch_dim], dtype="int32")'),
            "dim_encoder_time": CodeWrapper('Dim(dimension=tensor_encoder_time, name="encoder_time")'),
            "dim_encoder_feature": CodeWrapper('Dim(dimension=1024, name="encoder_feature")'),  # TODO: set dynamically
            "tensor_suffix_time": CodeWrapper('Tensor(name="suffix_time", dims=[batch_dim], dtype="int32")'),
            "dim_suffix_time": CodeWrapper('Dim(dimension=tensor_suffix_time, name="suffix_time")'),
            "dim_num_heads": CodeWrapper('Dim(name="num_heads", dimension=2)'),  # TODO: set dynamically
            "dim_head": CodeWrapper('Dim(name="head", dimension=64)'),  # TODO: set dynamically
            "dim_time": CodeWrapper("dim_initial_time + dim_encoder_time + dim_suffix_time"),
            "extern_data": extern_data,
            "model_outputs": model_outputs,
        },
        input_names=input_names,
        output_names=output_names,
        metadata=metadata,
    )


def export_step_model(model_kwargs: dict, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = Collection(
        [
            Import(f"{SpeechLmStep.__module__}.{SpeechLmStep.__name__}"),
            PyTorchModel(
                model_class_name=SpeechLmStep.__name__,
                model_kwargs=model_kwargs,
            ),
        ]
    )

    extern_data = {}
    input_names = []
    model_outputs = {
        "scores": {
            "dim_tags": CodeWrapper("(batch_dim, dim_token_time, dim_vocab)"),
            "dtype": "float32",
        }
    }
    output_names = ["scores"]
    metadata = {}

    for layer in range(24):  # TODO: set dynamically
        extern_data[f"state_l{layer:03d}_k_in"] = {
            "dim_tags": CodeWrapper("(batch_dim, dim_num_heads, dim_prefix_time, dim_head)"),
            "dtype": "float32",
        }
        extern_data[f"state_l{layer:03d}_v_in"] = {
            "dim_tags": CodeWrapper("(batch_dim, dim_num_heads, dim_prefix_time, dim_head)"),
            "dtype": "float32",
        }

        input_names.append(f"state_l{layer:03d}_k_in")
        if layer == 0:
            input_names.append(f"state_l{layer:03d}_k_in:size2")
        input_names.append(f"state_l{layer:03d}_v_in")

        model_outputs[f"state_l{layer:03d}_k_out"] = {
            "dim_tags": CodeWrapper("(batch_dim, dim_num_heads, dim_token_time, dim_head)"),
            "dtype": "float32",
        }
        model_outputs[f"state_l{layer:03d}_v_out"] = {
            "dim_tags": CodeWrapper("(batch_dim, dim_num_heads, dim_token_time, dim_head)"),
            "dtype": "float32",
        }

        output_names += [
            f"state_l{layer:03d}_k_out",
            f"state_l{layer:03d}_v_out",
        ]

        metadata[f"STATE_state_l{layer:03d}_k_in"] = f"state_l{layer:03d}_k_out"
        metadata[f"STATE_state_l{layer:03d}_v_in"] = f"state_l{layer:03d}_v_out"

    extern_data["token"] = {
        "dim_tags": CodeWrapper("(batch_dim, dim_token_time)"),
        "sparse_dim": CodeWrapper("dim_vocab"),
        "dtype": "int32",
    }
    input_names += ["token"]

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(f"{_forward_step.__module__}.{_forward_step.__name__}", import_as="forward_step"),
        checkpoint=checkpoint,
        returnn_config_dict={
            "dim_vocab": CodeWrapper('Dim(dimension=151936, name="vocab")'),  # TODO: set dynamically
            "dim_token_time": CodeWrapper('Dim(dimension=1, name="token_time")'),
            "tensor_prefix_time": CodeWrapper('Tensor(name="prefix_time", dims=[batch_dim], dtype="int32")'),
            "dim_prefix_time": CodeWrapper('Dim(dimension=tensor_prefix_time, name="prefix_time")'),
            "dim_num_heads": CodeWrapper('Dim(name="num_heads", dimension=2)'),  # TODO: set dynamically
            "dim_head": CodeWrapper('Dim(name="head", dimension=64)'),  # TODO: set dynamically
            "extern_data": extern_data,
            "model_outputs": model_outputs,
        },
        input_names=input_names,
        output_names=output_names,
        metadata=metadata,
    )


def export_ctc_scorer(model_kwargs: dict, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = Collection(
        [
            Import(f"{SpeechLmCtc.__module__}.{SpeechLmCtc.__name__}"),
            PyTorchModel(
                model_class_name=SpeechLmCtc.__name__,
                model_kwargs=model_kwargs,
            ),
        ]
    )

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_ctc_scorer_forward_step.__module__}.{_ctc_scorer_forward_step.__name__}", import_as="forward_step"
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "dim_enc_feature": CodeWrapper('Dim(dimension=1024, description="enc_feature")'),  # TODO: set dynamically
            "dim_target": CodeWrapper('Dim(dimension=151937, description="target")'),  # TODO: set dynamically
            "extern_data": {
                "encoder_state": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_enc_feature)"),
                    "dtype": "float32",
                },
            },
            "model_outputs": {
                "scores": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_target)"),
                    "dtype": "float32",
                },
            },
        },
        input_names=["encoder_state"],
        output_names=["scores"],
    )


def _forward_step_initializer(*, model: SpeechLmInitializer, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    initial_prompt = extern_data["initial_prompt"].raw_tensor
    assert initial_prompt is not None

    assert extern_data["initial_prompt"].dims[1].dyn_size_ext is not None
    initial_prompt_size = extern_data["initial_prompt"].dims[1].dyn_size_ext.raw_tensor  # [B]
    assert initial_prompt_size is not None

    encoder_states = extern_data["encoder_states"].raw_tensor
    assert encoder_states is not None

    assert extern_data["encoder_states"].dims[1].dyn_size_ext is not None
    encoder_states_size = extern_data["encoder_states"].dims[1].dyn_size_ext.raw_tensor  # [B]
    assert encoder_states_size is not None

    suffix_prompt = extern_data["suffix_prompt"].raw_tensor
    assert suffix_prompt is not None

    assert extern_data["suffix_prompt"].dims[1].dyn_size_ext is not None
    suffix_prompt_size = extern_data["suffix_prompt"].dims[1].dyn_size_ext.raw_tensor  # [B]
    assert suffix_prompt_size is not None

    scores, *out_states = model(
        initial_prompt=initial_prompt, encoder_states=encoder_states, suffix_prompt=suffix_prompt
    )  # [B, N, V]

    run_ctx.mark_as_output(name="scores", tensor=scores)

    expected = run_ctx.expected_outputs
    assert expected is not None

    idx = 0
    for layer in range(model.decoder.qwen_config.num_hidden_layers):
        assert expected[f"state_l{layer:03d}_k_out"].dims[2].dyn_size_ext is not None
        expected[f"state_l{layer:03d}_k_out"].dims[2].dyn_size_ext.raw_tensor = (
            initial_prompt_size + encoder_states_size + suffix_prompt_size
        )
        assert expected[f"state_l{layer:03d}_v_out"].dims[2].dyn_size_ext is not None
        expected[f"state_l{layer:03d}_v_out"].dims[2].dyn_size_ext.raw_tensor = (
            initial_prompt_size + encoder_states_size + suffix_prompt_size
        )

        run_ctx.mark_as_output(name=f"state_l{layer:03d}_k_out", tensor=out_states[idx])
        run_ctx.mark_as_output(name=f"state_l{layer:03d}_v_out", tensor=out_states[idx + 1])

        idx += 2


def _forward_step(*, model: SpeechLmStep, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    token = extern_data["token"].raw_tensor
    assert token is not None

    assert extern_data["state_l000_k_in"].dims[2].dyn_size_ext is not None
    prefix_lens = extern_data["state_l000_k_in"].dims[2].dyn_size_ext.raw_tensor  # [B]
    assert prefix_lens is not None

    kv_cache = []
    for layer in range(model.decoder.qwen_config.num_hidden_layers):
        kv_cache.append(extern_data[f"state_l{layer:03d}_k_in"].raw_tensor)
        kv_cache.append(extern_data[f"state_l{layer:03d}_v_in"].raw_tensor)

    scores, *new_kv_cache = model(token, prefix_lens.long(), *kv_cache)

    run_ctx.mark_as_output(name="scores", tensor=scores)

    idx = 0
    for layer in range(model.decoder.qwen_config.num_hidden_layers):
        run_ctx.mark_as_output(name=f"state_l{layer:03d}_k_out", tensor=new_kv_cache[idx])
        run_ctx.mark_as_output(name=f"state_l{layer:03d}_v_out", tensor=new_kv_cache[idx + 1])

        idx += 2


def _ctc_scorer_forward_step(*, model: SpeechLmCtc, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    encoder_state = extern_data["encoder_state"].raw_tensor
    assert encoder_state is not None

    scores = model(encoder_state=encoder_state)

    run_ctx.mark_as_output(name="scores", tensor=scores)
