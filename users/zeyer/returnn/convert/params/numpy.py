import numpy


def convert_tf_conv_to_pt_conv_filter(tf_filter: numpy.ndarray) -> numpy.ndarray:
    # in: (*filter_size, in_dim, out_dim)
    # out: (out_dim, in_dim, *filter_size)
    assert tf_filter.ndim >= 3
    n = tf_filter.ndim
    return tf_filter.transpose([n - 1, n - 2] + list(range(n - 2)))


def convert_tf_lstm_to_native_lstm_ff(old_w_ff_re: numpy.ndarray) -> numpy.ndarray:
    # in: (in_dim+dim, 4*dim)
    # out: (4*dim, in_dim)
    # See CustomCheckpointLoader MakeLoadBasicToNativeLstm.
    assert old_w_ff_re.ndim == 2
    assert old_w_ff_re.shape[1] % 4 == 0
    n_out = old_w_ff_re.shape[1] // 4
    assert old_w_ff_re.shape[0] > n_out
    n_in = old_w_ff_re.shape[0] - n_out
    old_w_ff, _ = numpy.split(old_w_ff_re, [n_in], axis=0)  # (in_dim,4*dim)
    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    # BasicLSTM: ijfo; Input: [inputs, h]
    # NativeLstm2: jifo
    old_w_ff_i, old_w_ff_j, old_w_ff_f, old_w_ff_o = numpy.split(old_w_ff, 4, axis=1)
    new_w_ff = numpy.concatenate([old_w_ff_j, old_w_ff_i, old_w_ff_f, old_w_ff_o], axis=1)  # (in_dim,4*dim)
    new_w_ff = new_w_ff.transpose()  # (4*dim,in_dim)
    return new_w_ff


def convert_tf_lstm_to_native_lstm_rec(old_w_ff_re: numpy.ndarray) -> numpy.ndarray:
    # in: (in_dim+dim, 4*dim)
    # out: (4*dim, in_dim)
    # See CustomCheckpointLoader MakeLoadBasicToNativeLstm.
    assert old_w_ff_re.ndim == 2
    assert old_w_ff_re.shape[1] % 4 == 0
    n_out = old_w_ff_re.shape[1] // 4
    assert old_w_ff_re.shape[0] > n_out
    n_in = old_w_ff_re.shape[0] - n_out
    _, old_w_rec = numpy.split(old_w_ff_re, [n_in], axis=0)  # (dim,4*dim)
    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    # BasicLSTM: ijfo; Input: [inputs, h]
    # NativeLstm2: jifo
    old_w_rec_i, old_w_rec_j, old_w_rec_f, old_w_rec_o = numpy.split(old_w_rec, 4, axis=1)
    new_w_rec = numpy.concatenate([old_w_rec_j, old_w_rec_i, old_w_rec_f, old_w_rec_o], axis=1)  # (dim,4*dim)
    new_w_rec = new_w_rec.transpose()  # (4*dim,dim)
    return new_w_rec


def convert_tf_lstm_to_native_lstm_bias(old_bias: numpy.ndarray, *, forget_gate_bias: float = 1.0) -> numpy.ndarray:
    # in: (4*dim,)
    # out: (4*dim,)
    # See CustomCheckpointLoader MakeLoadBasicToNativeLstm.
    assert old_bias.ndim == 1
    assert old_bias.shape[0] % 4 == 0
    n_out = old_bias.shape[0] // 4
    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    # BasicLSTM: ijfo; Input: [inputs, h]
    # NativeLstm2: jifo
    old_bias_i, old_bias_j, old_bias_f, old_bias_o = numpy.split(old_bias, 4, axis=0)
    old_bias_f += forget_gate_bias
    new_bias = numpy.concatenate([old_bias_j, old_bias_i, old_bias_f, old_bias_o], axis=0)  # (4*dim,)
    return new_bias
