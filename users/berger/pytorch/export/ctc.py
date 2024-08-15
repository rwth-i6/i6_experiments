import torch


def export(*, model: torch.nn.Module, f: str, in_dim: int):
    dummy_data = torch.randn(1, 30, in_dim, device="cpu")
    dummy_data_len = torch.tensor([30], dtype=torch.int32, device="cpu")

    torch.onnx.export(
        model=model.eval(),
        args=(dummy_data, dummy_data_len),
        f=f,
        verbose=True,
        input_names=["data", "data:size1"],
        output_names=["log_probs"],
        opset_version=17,
        dynamic_axes={
            # dict value: manually named axes
            "data": {0: "batch", 1: "time"},
            "data:size1": {0: "batch"},
            "log_probs": {0: "batch", 1: "time"},
        },
    )
