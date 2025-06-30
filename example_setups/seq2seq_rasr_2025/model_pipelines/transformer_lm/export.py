__all__ = ["export_model"]

import torch
from i6_core.returnn import PtCheckpoint
from returnn.tensor.tensor_dict import TensorDict
from sisyphus import tk

from i6_experiments.common.setups.serialization import Import

from ..common.onnx_export import export_model as _export_model
from ..common.pytorch_modules import lengths_to_padding_mask
from ..common.serializers import get_model_serializers
from .pytorch_modules import TransformerLm, TransformerLmConfig


def _forward_step(*, model: TransformerLm, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    tokens = extern_data["tokens"].raw_tensor
    assert isinstance(tokens, torch.Tensor)
    assert extern_data["tokens"].dims[1].dyn_size_ext is not None
    tokens_size = extern_data["tokens"].dims[1].dyn_size_ext.raw_tensor  # [B]
    assert tokens_size is not None

    seq_mask = lengths_to_padding_mask(tokens_size)

    logits = model.forward(input=tokens, seq_mask=seq_mask)

    # gather last token logits
    batch_size = logits.size(0)
    last_logits = logits[torch.arange(batch_size), tokens_size.long() - 1]
    scores = -torch.nn.functional.log_softmax(last_logits, dim=-1)

    run_ctx.mark_as_output(name="scores", tensor=scores)


def export_model(model_config: TransformerLmConfig, checkpoint: PtCheckpoint) -> tk.Path:
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
