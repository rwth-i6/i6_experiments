import numpy
import returnn.frontend as rf
from tensorflow.python.training.py_checkpoint_reader import CheckpointReader


def convert_tf_batch_norm_to_rf(
    *,
    reader: CheckpointReader,
    rf_name: str,
    rf_prefix_name: str,
    tf_prefix_name: str,
    var: rf.Parameter,
) -> numpy.ndarray:
    assert isinstance(reader, CheckpointReader)
    assert rf_name.startswith(rf_prefix_name)
    rf_suffix = rf_name[len(rf_prefix_name) :]
    tf_suffix = {"running_mean": "mean", "running_variance": "variance", "gamma": "gamma", "beta": "beta"}[rf_suffix]

    # TF model with earlier BN versions has strange naming
    tf_var_names = [
        name
        for name in reader.get_variable_to_shape_map()
        if name.startswith(tf_prefix_name) and name.endswith("_" + tf_suffix)
    ]
    assert len(tf_var_names) == 1, f"found {tf_var_names} for {rf_name}"
    value = reader.get_tensor(tf_var_names[0])
    assert var.batch_ndim == 1
    value = numpy.squeeze(value)
    assert value.ndim == 1 and value.shape == var.batch_shape
    return value
