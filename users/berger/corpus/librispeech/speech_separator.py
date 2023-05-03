import numpy as np

from sisyphus import tk

from i6_experiments.users.berger.recipe.returnn.conversion import (
    ConvertPytorchToReturnnJob,
)

speech_sep_checkpoint = tk.Path(
    "/work/asr3/converse/dependencies/sms_librispeech_84_281244/ckpt_281244.pth",
)

model_func = """
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
        F=513,
        recurrent_layers=3,
        units=600,
        K=2,
        dropout_input=0.,
        dropout_hidden=0.,
        dropout_linear=0.,
        output_activation='sigmoid')
    if checkpoint is not None:
        model.load_checkpoint(checkpoint, 'model.mask_net')

    with torch.no_grad():
        masks = model({"Y_abs": inputs})
        return masks
"""


def get_separator(
    python_exe: tk.Path,
    pytorch_to_returnn_root: tk.Path,
    returnn_root: tk.Path,
    padertorch_root: tk.Path,
):
    stft_abs = np.random.uniform(size=(1, 10, 513)).astype("float32")
    converter_job = ConvertPytorchToReturnnJob(
        pytorch_config=speech_sep_checkpoint,
        model_func=model_func,
        input=stft_abs,
        device="cpu",
        converter_kwargs={
            "inputs_data_kwargs": {
                "shape": (None, stft_abs.shape[2]),
                "batch_dim_axis": 0,
                "time_dim_axis": 1,
                "feature_dim_axis": 2,
            },
        },
        conversion_python_exe=python_exe,
        pytorch_to_returnn_root=pytorch_to_returnn_root,
        returnn_root=returnn_root,
        fairseq_root=padertorch_root,
    )
    return converter_job.out_returnn_model_dict, converter_job.out_returnn_checkpoint
