"""
Flashlight decoding
"""


def model_recog(
    *,
    model: Any,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    # TODO ...

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog: RecogDef[Any]
model_recog.output_with_beam = True
model_recog.output_blank_label = None
model_recog.batch_size_dependent = False  # not totally correct, but we treat it as such...
