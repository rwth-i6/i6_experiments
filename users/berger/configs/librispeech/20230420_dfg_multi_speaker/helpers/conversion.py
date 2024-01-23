def convert_pytorch_model_to_returnn(wrapped_import, inputs: torch.Tensor, checkpoint):
    import typing

    if typing.TYPE_CHECKING or not wrapped_import:
        import torch
        import einops
        from padertorch.contrib.examples.source_separation.tasnet import tas_coders
        from padertorch.contrib.examples.source_separation.pit import model as pit_model
    else:
        torch = wrapped_import("torch")
        padertorch = wrapped_import("padertorch")
        einops = wrapped_import("einops")
        pit_model = wrapped_import("padertorch.contrib.examples.source_separation.pit.model")

    model = pit_model.PermutationInvariantTrainingModel(
        F=257,
        recurrent_layers=3,
        units=600,
        K=2,
        dropout_input=0.0,
        dropout_hidden=0.0,
        dropout_linear=0.0,
        output_activation="sigmoid",
    )
    if checkpoint is not None:
        model.load_checkpoint(checkpoint, "model.mask_net")

    with torch.no_grad():
        masks = model({"Y_abs": inputs})
        return masks
