import numpy
from tensorflow.python.training.py_checkpoint_reader import CheckpointReader
import torch


def convert_tf_batch_norm_to_pt(
    *,
    reader: CheckpointReader,
    pt_name: str,
    pt_prefix_name: str,
    tf_prefix_name: str,
    var: torch.nn.Parameter,
) -> numpy.ndarray:
    assert isinstance(reader, CheckpointReader)
    assert pt_name.startswith(pt_prefix_name)
    pt_suffix = pt_name[len(pt_prefix_name) :]

    if pt_suffix == "num_batches_tracked":
        assert reader.has_tensor("global_step")
        global_step = reader.get_tensor("global_step")
        assert isinstance(global_step, numpy.int64)
        return numpy.array(global_step)

    tf_suffix = {"running_mean": "mean", "running_var": "variance", "weight": "gamma", "bias": "beta"}[pt_suffix]

    # TF model with earlier BN versions has strange naming
    tf_var_names = [
        name
        for name in reader.get_variable_to_shape_map()
        if name.startswith(tf_prefix_name) and name.endswith("_" + tf_suffix)
    ]
    assert len(tf_var_names) == 1, f"found {tf_var_names} for {pt_name}"
    value = reader.get_tensor(tf_var_names[0])
    assert len(var.shape) == 1
    value = numpy.squeeze(value)
    assert value.ndim == 1 and value.shape == var.shape
    return value
