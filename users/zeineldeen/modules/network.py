"""
RETURNN network dict creation helper
"""
from typing import Optional, List, Union, Tuple, Any


class ReturnnNetwork:
    """
    Represents a generic RETURNN network
    see docs: https://returnn.readthedocs.io/en/latest/
    """

    def __init__(self):
        self._net = {}

    def get_net(self):
        return self._net

    def add_copy_layer(self, name, source, **kwargs):
        self._net[name] = {"class": "copy", "from": source}
        self._net[name].update(kwargs)
        return name

    def add_eval_layer(self, name, source, eval, **kwargs):
        self._net[name] = {"class": "eval", "eval": eval, "from": source}
        self._net[name].update(kwargs)
        return name

    def add_split_dim_layer(self, name, source, axis="F", dims=(-1, 1), **kwargs):
        self._net[name] = {"class": "split_dims", "axis": axis, "dims": dims, "from": source}
        self._net[name].update(kwargs)
        return name

    def add_conv_layer(
        self,
        name,
        source,
        filter_size,
        n_out,
        l2,
        padding="same",
        activation=None,
        with_bias=True,
        forward_weights_init=None,
        strides=None,
        param_variational_noise=None,
        param_dropout=None,
        param_dropout_min_ndim=None,
        **kwargs,
    ):
        d = {
            "class": "conv",
            "from": source,
            "padding": padding,
            "filter_size": filter_size,
            "n_out": n_out,
            "activation": activation,
            "with_bias": with_bias,
        }
        if strides:
            d["strides"] = strides
        if l2:
            d["L2"] = l2
        if forward_weights_init:
            d["forward_weights_init"] = forward_weights_init
        if param_variational_noise:
            d["param_variational_noise"] = param_variational_noise
        if param_dropout:
            d["param_dropout"] = param_dropout
            if param_dropout_min_ndim is not None:
                d["param_dropout_min_ndim"] = param_dropout_min_ndim
        d.update(kwargs)
        self._net[name] = d
        return name

    def add_linear_layer(
        self,
        name,
        source,
        n_out=None,
        activation=None,
        with_bias=True,
        dropout=0.0,
        l2=0.0,
        forward_weights_init=None,
        param_dropout=None,
        param_dropout_min_ndim=None,
        param_variational_noise=None,
        **kwargs,
    ):
        d = {"class": "linear", "activation": activation, "with_bias": with_bias, "from": source}
        if n_out:
            d["n_out"] = n_out
        else:
            assert "target" in kwargs, "target must be specified to define output dimension"
        if dropout:
            d["dropout"] = dropout
        if l2:
            d["L2"] = l2
        if forward_weights_init:
            d["forward_weights_init"] = forward_weights_init
        if param_dropout:
            d["param_dropout"] = param_dropout
            if param_dropout_min_ndim is not None:
                d["param_dropout_min_ndim"] = param_dropout_min_ndim
        if param_variational_noise:
            d["param_variational_noise"] = param_variational_noise
        d.update(kwargs)
        self._net[name] = d
        return name

    def add_pool_layer(self, name, source, pool_size, mode="max", **kwargs):
        self._net[name] = {"class": "pool", "from": source, "pool_size": pool_size, "mode": mode, "trainable": False}
        self._net[name].update(kwargs)
        return name

    def add_merge_dims_layer(self, name, source, axes="static", **kwargs):
        self._net[name] = {"class": "merge_dims", "from": source, "axes": axes}
        self._net[name].update(kwargs)
        return name

    def add_rec_layer(
        self,
        name,
        source,
        n_out,
        l2,
        rec_weight_dropout=0.0,
        weights_init=None,
        direction=1,
        unit="nativelstm2",
        **kwargs,
    ):
        d = {"class": "rec", "unit": unit, "n_out": n_out, "direction": direction, "from": source}
        if l2:
            d["L2"] = l2
        if rec_weight_dropout:
            d.setdefault("unit_opts", {}).update({"rec_weight_dropout": rec_weight_dropout})
        if weights_init:
            d.setdefault("unit_opts", {}).update(
                {"forward_weights_init": weights_init, "recurrent_weights_init": weights_init}
            )
        d.update(kwargs)
        self._net[name] = d
        return name

    def add_choice_layer(self, name, source, target, beam_size=12, initial_output=0, input_type=None, **kwargs):
        self._net[name] = {
            "class": "choice",
            "target": target,
            "beam_size": beam_size,
            "from": source,
            "initial_output": initial_output,
        }
        if input_type:
            self._net[name]["input_type"] = input_type
        self._net[name].update(kwargs)
        return name

    def add_compare_layer(self, name, source, value, kind="equal", **kwargs):
        self._net[name] = {"class": "compare", "kind": kind, "from": source, "value": value}
        self._net[name].update(kwargs)
        return name

    def add_cast_layer(self, name, source, dtype):
        self._net[name] = {"class": "cast", "from": source, "dtype": dtype}
        return name

    def add_combine_layer(self, name, source, kind, n_out=None, **kwargs):
        self._net[name] = {"class": "combine", "kind": kind, "from": source}
        if n_out is not None:
            self._net[name]["n_out"] = n_out
        self._net[name].update(kwargs)
        return name

    def add_activation_layer(self, name, source, activation, **kwargs):
        self._net[name] = {"class": "activation", "activation": activation, "from": source}
        self._net[name].update(kwargs)
        return name

    def add_softmax_over_spatial_layer(self, name, source, **kwargs):
        self._net[name] = {"class": "softmax_over_spatial", "from": source}
        self._net[name].update(kwargs)
        return name

    def add_generic_att_layer(self, name, weights, base, **kwargs):
        self._net[name] = {"class": "generic_attention", "weights": weights, "base": base}
        self._net[name].update(kwargs)
        return name

    def add_squeeze_layer(self, name, source, **kwargs):
        self._net[name] = {"class": "squeeze", "from": source, **kwargs}
        return name

    def add_rnn_cell_layer(
        self, name, source, n_out, unit="LSTMBlock", l2=0.0, unit_opts=None, weights_init=None, **kwargs
    ):
        d = {"class": "rnn_cell", "unit": unit, "n_out": n_out, "from": source}
        if l2:
            d["L2"] = l2
        if unit_opts:
            d["unit_opts"] = unit_opts
        if weights_init:
            d["weights_init"] = weights_init
        d.update(kwargs)
        self._net[name] = d
        return name

    def add_softmax_layer(
        self,
        name,
        source,
        l2=None,
        loss=None,
        target=None,
        dropout=0.0,
        loss_opts=None,
        forward_weights_init=None,
        loss_scale=None,
        param_dropout=None,
        param_dropout_min_ndim=None,
        param_variational_noise=None,
        **kwargs,
    ):
        d = {"class": "softmax", "from": source}
        if dropout:
            d["dropout"] = dropout
        if target:
            d["target"] = target
        if loss:
            d["loss"] = loss
            if loss_opts:
                d["loss_opts"] = loss_opts
        if l2:
            d["L2"] = l2
        if forward_weights_init:
            d["forward_weights_init"] = forward_weights_init
        if param_dropout:
            d["param_dropout"] = param_dropout
            if param_dropout_min_ndim is not None:
                d["param_dropout_min_ndim"] = param_dropout_min_ndim
        if param_variational_noise:
            d["param_variational_noise"] = param_variational_noise
        if loss_scale:
            d["loss_scale"] = loss_scale
        d.update(kwargs)
        self._net[name] = d
        return name

    def add_dropout_layer(self, name, source, dropout, dropout_noise_shape=None, **kwargs):
        self._net[name] = {"class": "dropout", "from": source, "dropout": dropout}
        if dropout_noise_shape:
            self._net[name]["dropout_noise_shape"] = dropout_noise_shape
        self._net[name].update(kwargs)
        return name

    def add_reduceout_layer(self, name, source, num_pieces=2, mode="max", **kwargs):
        self._net[name] = {"class": "reduce_out", "from": source, "num_pieces": num_pieces, "mode": mode}
        self._net[name].update(kwargs)
        return name

    def add_subnet_rec_layer(self, name, unit, target, source=None, include_eos=False, max_seq_len=None, **kwargs):
        if source is None:
            source = []
        self._net[name] = {
            "class": "rec",
            "from": source,
            "unit": unit,
            "target": target,
        }
        if max_seq_len is None:
            self._net[name]["max_seq_len"] = "max_len_from('base:encoder')"
        else:
            self._net[name]["max_seq_len"] = max_seq_len
        if include_eos:
            self._net[name]["include_eos"] = include_eos
        self._net[name].update(kwargs)
        return name

    def add_decide_layer(self, name, source, target, loss="edit_distance", **kwargs):
        self._net[name] = {"class": "decide", "from": source, "loss": loss, "target": target}
        self._net[name].update(kwargs)
        return name

    def add_slice_layer(self, name, source, axis, **kwargs):
        self._net[name] = {"class": "slice", "from": source, "axis": axis, **kwargs}
        return name

    def add_subnetwork(self, name, source, subnetwork_net, **kwargs):
        self._net[name] = {"class": "subnetwork", "from": source, "subnetwork": subnetwork_net, **kwargs}
        return name

    def add_layer_norm_layer(self, name, source, **kwargs):
        self._net[name] = {"class": "layer_norm", "from": source, **kwargs}
        return name

    def add_batch_norm_layer(self, name, source, opts=None, **kwargs):
        self._net[name] = {"class": "batch_norm", "from": source, **kwargs}
        if opts:
            assert isinstance(opts, dict)
            self._net[name].update(opts)
        return name

    def add_self_att_layer(
        self,
        name,
        source,
        n_out,
        num_heads,
        total_key_dim,
        att_dropout=0.0,
        key_shift=None,
        forward_weights_init=None,
        l2=0.0,
        attention_left_only=False,
        param_variational_noise=None,
        param_dropout=None,
        param_dropout_min_ndim=None,
        **kwargs,
    ):
        d = {
            "class": "self_attention",
            "from": source,
            "n_out": n_out,
            "num_heads": num_heads,
            "total_key_dim": total_key_dim,
        }
        if att_dropout:
            d["attention_dropout"] = att_dropout
        if key_shift:
            d["key_shift"] = key_shift
        if forward_weights_init:
            d["forward_weights_init"] = forward_weights_init
        if l2:
            d["L2"] = l2
        if attention_left_only:
            d["attention_left_only"] = attention_left_only
        if param_dropout:
            d["param_dropout"] = param_dropout
            if param_dropout_min_ndim is not None:
                d["param_dropout_min_ndim"] = param_dropout_min_ndim
        if param_variational_noise:
            d["param_variational_noise"] = param_variational_noise
        d.update(kwargs)
        self._net[name] = d
        return name

    def add_pos_encoding_layer(self, name, source, add_to_input=True, **kwargs):
        self._net[name] = {"class": "positional_encoding", "from": source, "add_to_input": add_to_input}
        self._net[name].update(kwargs)
        return name

    def add_relative_pos_encoding_layer(self, name, source, n_out, forward_weights_init=None, **kwargs):
        self._net[name] = {"class": "relative_positional_encoding", "from": source, "n_out": n_out}
        if forward_weights_init:
            self._net[name]["forward_weights_init"] = forward_weights_init
        self._net[name].update(kwargs)
        return name

    def add_constant_layer(self, name, value, **kwargs):
        self._net[name] = {"class": "constant", "value": value}
        self._net[name].update(kwargs)
        return name

    def add_gating_layer(self, name, source, activation="identity", **kwargs):
        """
        out = activation(a) * gate_activation(b)  (gate_activation is sigmoid by default)
        In case of one source input, it will split by 2 over the feature dimension
        """
        self._net[name] = {"class": "gating", "from": source, "activation": activation}
        self._net[name].update(kwargs)
        return name

    def add_pad_layer(self, name, source, axes, padding, **kwargs):
        self._net[name] = {"class": "pad", "from": source, "axes": axes, "padding": padding}
        self._net[name].update(**kwargs)
        return name

    def add_reduce_layer(self, name, source, mode, axes, keep_dims=False, **kwargs):
        self._net[name] = {"class": "reduce", "from": source, "mode": mode, "axes": axes, "keep_dims": keep_dims}
        self._net[name].update(**kwargs)
        return name

    def add_variable_layer(self, name, shape, **kwargs):
        self._net[name] = {"class": "variable", "shape": shape}
        self._net[name].update(kwargs)
        return name

    def add_switch_layer(self, name, condition, true_from, false_from, **kwargs):
        self._net[name] = {"class": "switch", "condition": condition, "true_from": true_from, "false_from": false_from}
        self._net[name].update(kwargs)
        return name

    def add_kenlm_layer(self, name, lm_file, **kwargs):
        self._net[name] = {"class": "kenlm", "lm_file": lm_file, **kwargs}
        return name

    def add_length_layer(self, name, source, **kwargs):
        self._net[name] = {"class": "length", "from": source, **kwargs}
        return name

    def add_reinterpret_data_layer(self, name, source, **kwargs):
        self._net[name] = {"class": "reinterpret_data", "from": source, **kwargs}
        return name

    def add_masked_computation_layer(self, name, source, mask, unit, **kwargs):
        self._net[name] = {"class": "masked_computation", "from": source, "mask": mask, "unit": unit, **kwargs}
        return name

    def add_unmask_layer(self, name, source, mask, **kwargs):
        self._net[name] = {"class": "unmask", "from": source, "mask": mask, **kwargs}
        return name

    def add_window_layer(self, name, source, window_size, **kwargs):
        self._net[name] = {"class": "window", "from": source, "window_size": window_size, **kwargs}
        return name

    def add_range_layer(self, name, limit, **kwargs):
        self._net[name] = {"class": "range", "limit": limit, **kwargs}
        return name

    def add_range_in_axis_layer(self, name, source, axis, **kwargs):
        self._net[name] = {"class": "range_in_axis", "from": source, "axis": axis, **kwargs}
        return name

    def add_conv_block(
        self,
        name,
        source,
        hwpc_sizes,
        l2,
        activation,
        dropout=0.0,
        init=None,
        use_striding=False,
        split_input=True,
        merge_out=True,
        merge_out_fixed=False,
        spatial_dims=None,
        prefix_name=None,
        param_variational_noise=None,
        param_dropout=None,
        param_dropout_min_ndim=None,
    ):
        if split_input:
            src = self.add_split_dim_layer("source0", source)
        else:
            src = source
        if prefix_name is None:
            prefix_name = ""
        out_dim = None
        for idx, hwpc in enumerate(hwpc_sizes):
            filter_size, pool_size, n_out = hwpc
            extra_conv_opts = {}
            if spatial_dims:
                extra_conv_opts["in_spatial_dims"] = spatial_dims
            if spatial_dims or merge_out_fixed:
                from returnn.tensor import Dim

                spatial_dims = [Dim(None, name=f"{prefix_name}conv{idx}.{i}") for i in range(len(filter_size))]
                extra_conv_opts["out_spatial_dims"] = spatial_dims
            if merge_out_fixed:
                from returnn.tensor import Dim

                out_dim = Dim(n_out, name=f"{prefix_name}conv{idx}.out")
                extra_conv_opts["out_dim"] = out_dim
            src = self.add_conv_layer(
                f"{prefix_name}conv%i" % idx,
                src,
                filter_size=filter_size,
                n_out=n_out,
                l2=l2,
                activation=activation,
                forward_weights_init=init,
                strides=pool_size if use_striding else None,
                param_variational_noise=param_variational_noise,
                param_dropout=param_dropout,
                param_dropout_min_ndim=param_dropout_min_ndim,
                **extra_conv_opts,
            )
            if pool_size and not use_striding:
                src = self.add_pool_layer(f"{prefix_name}conv%ip" % idx, src, pool_size=pool_size, padding="same")
        if dropout:
            src = self.add_dropout_layer(f"{prefix_name}conv_dropout", src, dropout=dropout)
        if merge_out:
            if merge_out_fixed:
                assert spatial_dims and out_dim
                return self.add_merge_dims_layer(name, src, axes=spatial_dims[1:] + [out_dim], keep_order=True)
            else:
                return self.add_merge_dims_layer(name, src)
        return self.add_copy_layer(name, src)

    def add_lstm_layers(self, input, num_layers, lstm_dim, dropout, l2, rec_weight_dropout, pool_sizes, bidirectional):
        src = input
        pool_idx = 0
        for layer in range(num_layers):
            lstm_fw_name = self.add_rec_layer(
                name="lstm%i_fw" % layer,
                source=src,
                n_out=lstm_dim,
                direction=1,
                dropout=dropout,
                l2=l2,
                rec_weight_dropout=rec_weight_dropout,
            )
            if bidirectional:
                lstm_bw_name = self.add_rec_layer(
                    name="lstm%i_bw" % layer,
                    source=src,
                    n_out=lstm_dim,
                    direction=-1,
                    dropout=dropout,
                    l2=l2,
                    rec_weight_dropout=rec_weight_dropout,
                )
                src = [lstm_fw_name, lstm_bw_name]
            else:
                src = lstm_fw_name
            if pool_sizes and pool_idx < len(pool_sizes):
                lstm_pool_name = "lstm%i_pool" % layer
                src = self.add_pool_layer(
                    name=lstm_pool_name, source=src, pool_size=(pool_sizes[pool_idx],), padding="same"
                )
                pool_idx += 1
        return src

    def add_residual_lstm_layers(
        self,
        input,
        num_layers,
        lstm_dim,
        dropout,
        l2,
        rec_weight_dropout,
        pool_sizes,
        residual_proj_dim=None,
        batch_norm=True,
    ):
        src = input
        pool_idx = 0

        for layer in range(num_layers):
            # Forward LSTM
            lstm_fw_name = self.add_rec_layer(
                name="lstm%i_fw" % layer,
                source=src,
                n_out=lstm_dim,
                direction=1,
                l2=l2,
                dropout=dropout,
                rec_weight_dropout=rec_weight_dropout,
            )

            # Backward LSTM
            lstm_bw_name = self.add_rec_layer(
                name="lstm%i_bw" % layer,
                source=src,
                n_out=lstm_dim,
                direction=-1,
                l2=l2,
                dropout=dropout,
                rec_weight_dropout=rec_weight_dropout,
            )

            # Concat LSTM outputs
            new_src = [lstm_fw_name, lstm_bw_name]

            # If given, project both LSTM output and LSTM input
            residual_lstm_out = new_src
            residual_lstm_in = src
            if residual_proj_dim:
                residual_lstm_out = self.add_linear_layer(
                    "lstm%i_lin_proj" % layer, new_src, n_out=residual_proj_dim, l2=l2
                )
                residual_lstm_in = self.add_linear_layer(
                    "lstm%i_inp_lin_proj" % layer, src, n_out=residual_proj_dim, l2=l2
                )

            # residual connection
            lstm_combine = self.add_combine_layer(
                "lstm%i_combine" % layer, [residual_lstm_in, residual_lstm_out], kind="add", n_out=residual_proj_dim
            )

            # apply batch norm if enabled
            if batch_norm:
                lstm_combine = self.add_batch_norm_layer(lstm_combine + "_bn", lstm_combine)

            if pool_sizes and pool_idx < len(pool_sizes):
                lstm_pool_name = "lstm%i_pool" % layer
                src = self.add_pool_layer(
                    name=lstm_pool_name, source=lstm_combine, pool_size=(pool_sizes[pool_idx],), padding="same"
                )
                pool_idx += 1
            else:
                src = lstm_combine

        return src

    def add_dot_layer(self, name, source, **kwargs):
        self._net[name] = {"class": "dot", "from": source}
        self._net[name].update(kwargs)
        return name

    def add_generic_layer(
        self, name: str, *, cls: str, source: Optional[Union[str, List[str], List[Tuple[str, Any]]]], **kwargs
    ):
        """
        :param name: layer name
        :param source: layer source
        :param cls: layer class
        :param kwargs: layer kwargs
        :return: layer name
        """
        self._net[name] = {"class": cls}
        if source:
            self._net[name]["from"] = source
        self._net[name].update(kwargs)
        return name

    def __setitem__(self, key, value):
        self._net[key] = value

    def __getitem__(self, item):
        return self._net[item]

    def __contains__(self, item):
        return item in self._net

    def update(self, d: dict):
        self._net.update(d)

    def __str__(self):
        """
        Only for debugging
        """
        res = "network = {\n"
        for k, v in self._net.items():
            res += "%s: %r\n" % (k, v)
        return res + "}"
