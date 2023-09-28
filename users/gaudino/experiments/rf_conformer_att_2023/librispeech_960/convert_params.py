import returnn.frontend as rf
from returnn.tensor import Tensor

def convert_tf_lstm_to_torch_lstm():
    # PyTorch (cuDNN) weights are in ifco order (?),
    # which we defined as the standard for the RF.
    # Also, they are (in_dim|out_dim, 4*out_dim).
    # RETURNN NativeLstm2 weight order: cell-in + input, forget and output gates (cifo).
    # And they are (4*out_dim, in_dim|out_dim).
    # So we need to reorder the params (ifco->cifo) and transpose them.
    # See also CustomCheckpointLoader and convert_cudnn_canonical_to_lstm_block.
    # TODO: ideally, we would create a new NativeLstm variant which just uses the same order.
    rec_weight = rec_weight.copy_transpose((out_dim, 4 * out_dim))
    ff_weight = ff_weight.copy_transpose((in_dim, 4 * out_dim))
    out_dim_ = out_dim.copy(same_as_self=False, description="(out-dim)")
    rec_weight_ = rf.split(rec_weight, axis=4 * out_dim, out_dims=[out_dim_] * 4)
    ff_weight_ = rf.split(ff_weight, axis=4 * out_dim, out_dims=[out_dim_] * 4)
    bias_ = rf.split(bias, axis=4 * out_dim, out_dims=[out_dim_] * 4)
    rec_weight, _ = rf.concat(
        (rec_weight_[2], out_dim_),
        (rec_weight_[0], out_dim_),
        (rec_weight_[1], out_dim_),
        (rec_weight_[3], out_dim_),
    )
    ff_weight, _ = rf.concat(
        (ff_weight_[2], out_dim_), (ff_weight_[0], out_dim_), (ff_weight_[1], out_dim_), (ff_weight_[3], out_dim_)
    )
    bias, _ = rf.concat((bias_[2], out_dim_), (bias_[0], out_dim_), (bias_[1], out_dim_), (bias_[3], out_dim_))
    rec_weight = Tensor(
        "rec_weight", [out_dim, 4 * out_dim], dtype=rec_weight.dtype, raw_tensor=rec_weight.raw_tensor
    )
    ff_weight = Tensor("ff_weight", [in_dim, 4 * out_dim], dtype=ff_weight.dtype, raw_tensor=ff_weight.raw_tensor)
    bias = Tensor("bias", [4 * out_dim], dtype=bias.dtype, raw_tensor=bias.raw_tensor)