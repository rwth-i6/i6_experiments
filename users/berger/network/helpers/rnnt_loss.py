from sisyphus.delayed_ops import DelayedFormat
from i6_core.tools.git import CloneGitRepositoryJob


def rnnt_loss(sources, blank_label=0):
    from returnn.extern.WarpRna import rna_loss

    logits = sources(0, as_data=True, auto_convert=False)
    targets = sources(1, as_data=True, auto_convert=False)
    encoder = sources(2, as_data=True, auto_convert=False)

    loss = rna_loss(
        logits.get_placeholder_as_batch_major(),
        targets.get_placeholder_as_batch_major(),
        encoder.get_sequence_lengths(),
        targets.get_sequence_lengths(),
        blank_label=blank_label,
    )
    loss.set_shape((None,))
    return loss


def add_rnnt_loss(
    network: dict,
    encoder: str,
    joint_output: str,
    targets: str,
    num_classes: int,
    blank_index: int = 0,
):
    network["output"] = {
        "class": "linear",
        "from": joint_output,
        "activation": "log_softmax",
        "n_out": num_classes,
    }

    network["rnnt_loss"] = {
        "class": "eval",
        "from": ["output", targets, encoder],
        "loss": "as_is",
        "out_type": {"batch_dim_axis": 0, "time_dim_axis": None, "shape": ()},
        "eval": f'self.network.get_config().typed_value("rnnt_loss")(source, {blank_index})',
    }

    return [rnnt_loss]


def rnnt_loss_compressed(sources, blank_label=0):
    from returnn.extern_private.BergerMonotonicRNNT import rnnt_loss

    logits = sources(0, as_data=True, auto_convert=False)
    targets = sources(1, as_data=True, auto_convert=False)
    encoder = sources(2, as_data=True, auto_convert=False)

    loss = rnnt_loss(
        logits.placeholder,
        targets.get_placeholder_as_batch_major(),
        encoder.get_sequence_lengths(),
        targets.get_sequence_lengths(),
        blank_label=blank_label,
        input_type="logit",
    )
    loss.set_shape((None,))
    return loss


def rnnt_loss_compressed_v2(sources, blank_label: int = 0):
    from tensorflow_binding.returnn_tf_op import monotonic_rnnt_loss

    logits = sources(0, as_data=True, auto_convert=False)
    targets = sources(1, as_data=True, auto_convert=False)
    encoder = sources(2, as_data=True, auto_convert=False)

    loss = monotonic_rnnt_loss(
        logits.placeholder,
        targets.get_placeholder_as_batch_major(),
        encoder.get_sequence_lengths(),
        targets.get_sequence_lengths(),
        blank_label=blank_label,
    )
    loss.set_shape((None,))
    return loss


def add_rnnt_loss_compressed(
    network: dict,
    encoder: str,
    joint_output: str,
    targets: str,
    num_classes: int,
    blank_index: int = 0,
    loss_v2: bool = False,
):
    network["output"] = {
        "class": "linear",
        "from": joint_output,
        "activation": None,
        "n_out": num_classes,
        "out_type": {
            "batch_dim_axis": None,
            "time_dim_axis": 0,
            "shape": (None, num_classes),
        },
    }

    network["rnnt_loss"] = {
        "class": "eval",
        "from": ["output", targets, encoder],
        "loss": "as_is",
        "out_type": {"batch_dim_axis": 0, "time_dim_axis": None, "shape": ()},
    }

    if loss_v2:
        network["rnnt_loss"][
            "eval"
        ] = f'self.network.get_config().typed_value("rnnt_loss_compressed_v2")(source, {blank_index})'
    else:
        network["rnnt_loss"][
            "eval"
        ] = f'self.network.get_config().typed_value("rnnt_loss_compressed")(source, {blank_index})'

    if loss_v2:
        repo = CloneGitRepositoryJob(
            "https://github.com/SimBe195/monotonic-rnnt.git", checkout_folder_name="monotonic-rnnt"
        ).out_repository
        return [
            "import sys",
            DelayedFormat('sys.path.append("{}")', repo),
            rnnt_loss_compressed_v2,
        ]
    else:
        return [rnnt_loss_compressed]
