import torch


def export(*, model: torch.nn.Module, model_filename: str, in_dim: int):
    dummy_data = torch.randn(1, 30, in_dim, device="cpu")
    torch.onnx.export(
        model=model.eval(),
        args=(dummy_data, None),
        f=model_filename,
        verbose=True,
        input_names=["data", "data_len"],
        output_names=["classes"],
        opset_version=17,
        dynamic_axes={
            # dict value: manually named axes
            "data": {0: "batch", 1: "time"},
            "data_len": {0: "batch"},
            "targets": {0: "batch", 1: "time"},
        },
    )
