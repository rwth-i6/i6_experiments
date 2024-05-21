from i6_experiments.users.zeineldeen.models.asr.encoder.conformer_encoder_v2 import ConformerEncoderV2


class EBranchformerEncoder(ConformerEncoderV2):
    """
    Implement E-branchformer Encoder Architecture
    * Ref: https://arxiv.org/pdf/2210.00077.pdf
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_conv_spatial_gating_unit(self, prefix_name, source, layer_index):
        # see also here: https://github.com/espnet/espnet/blob/master/espnet2/asr/layers/cgmlp.py#L15

        branch_a = self.network.add_slice_layer(
            "{}_branch_a".format(prefix_name), source, "F", slice_start=0, slice_end=self.enc_key_dim * 3
        )

        branch_b = self.network.add_slice_layer(
            "{}_branch_b".format(prefix_name), source, "F", slice_start=self.enc_key_dim * 3
        )

        br_part_b_ln = self.network.add_layer_norm_layer("{}_branch_b_ln".format(prefix_name), branch_b)

        br_part_b_depthwise_conv = self.network.add_conv_layer(
            "{}_branch_b_depthwise_conv".format(prefix_name),
            br_part_b_ln,
            n_out=self.enc_key_dim * 3,
            filter_size=(self.conv_kernel_size,),
            groups=self.enc_key_dim * 3,
            l2=self.l2,
            param_dropout=self.conv_weight_drop,
            param_dropout_min_ndim=2,
            param_variational_noise=self.conv_weight_noise,
        )

        br_merge = self.network.add_eval_layer(
            "{}_branch_merge".format(prefix_name), [branch_a, br_part_b_depthwise_conv], "source(0) * source(1)"
        )

        dropout = self.network.add_dropout_layer("{}_dropout".format(prefix_name), br_merge, dropout=self.dropout)

        return dropout

    def _create_conv_gating_mlp(self, prefix_name, source, layer_index):
        prefix_name = "{}_cgmlp".format(prefix_name)

        ln = self.network.add_layer_norm_layer("{}_ln".format(prefix_name), source)

        ff1 = self.network.add_linear_layer(
            "{}_ff_1".format(prefix_name),
            ln,
            n_out=6 * self.enc_key_dim,  # TODO: make it configurable
            l2=self.l2,
            forward_weights_init=self.ff_init,
            with_bias=False,
            param_dropout=self.ff_weight_drop,
            param_dropout_min_ndim=2,
            param_variational_noise=self.ff_weight_noise,
        )

        gelu_act = self.network.add_activation_layer("{}_gelu".format(prefix_name), ff1, activation="gelu")

        csgu = self._create_conv_spatial_gating_unit(prefix_name, gelu_act, layer_index)

        br_merge_ff = self.network.add_linear_layer(
            "{}_ff_2".format(prefix_name),
            csgu,
            n_out=self.enc_key_dim,
            l2=self.l2,
            forward_weights_init=self.ff_init,
            with_bias=False,
            param_dropout=self.ff_weight_drop,
            param_dropout_min_ndim=2,
            param_variational_noise=self.ff_weight_noise,
        )

        return br_merge_ff

    def _create_merge_module(self, prefix_name, *, source, global_extracter, local_extracter, layer_index):
        prefix_name = "{}_merge_module".format(prefix_name)

        # concat on feature dim
        glb_lcl_merge = self.network.add_copy_layer(
            "{}_global_local_merge".format(prefix_name), [global_extracter, local_extracter]
        )

        depthwise_conv = self.network.add_conv_layer(
            "{}_depthwise_conv".format(prefix_name),
            glb_lcl_merge,
            n_out=2 * self.enc_key_dim,
            filter_size=(self.conv_kernel_size,),
            groups=2 * self.enc_key_dim,
            l2=self.l2,
            param_dropout=self.conv_weight_drop,
            param_dropout_min_ndim=2,
            param_variational_noise=self.conv_weight_noise,
        )

        depthwise_conv_res = self.network.add_combine_layer(
            "{}_depthwise_conv_res".format(prefix_name),
            kind="add",
            source=[glb_lcl_merge, depthwise_conv],
            n_out=2 * self.enc_key_dim,
        )

        ff = self.network.add_linear_layer(
            "{}_ff".format(prefix_name),
            depthwise_conv_res,
            n_out=self.enc_key_dim,
            l2=self.l2,
            forward_weights_init=self.ff_init,
            with_bias=False,
            param_dropout=self.ff_weight_drop,
            param_dropout_min_ndim=2,
            param_variational_noise=self.ff_weight_noise,
        )

        dropout = self.network.add_dropout_layer("{}_dropout".format(prefix_name), ff, dropout=self.dropout)

        merge_mod_res = self.network.add_combine_layer(
            "{}_res".format(prefix_name), kind="add", source=[source, dropout], n_out=self.enc_key_dim
        )

        return merge_mod_res

    def _block_prefix_name(self, layer_index: int) -> str:
        assert layer_index >= 1
        if self.add_to_prefix_name:
            prefix_name = "ebranchformer_block_%s_%02i" % (self.add_to_prefix_name, layer_index)
        else:
            prefix_name = "ebranchformer_block_%02i" % layer_index
        return prefix_name

    def _create_conformer_block(self, i, source):
        """
        Create an ebranchformer block:

        FF -> [MHSA, Conv] -> Merger -> FF -> LN
        """

        prefix_name = self._block_prefix_name(i)

        ff_module1 = self._create_ff_module(prefix_name, 1, source, i)

        # create branch 1: MHSA
        mhsa = self._create_mhsa_module(prefix_name, ff_module1, i)

        # create branch 2: Convolutional gating MLP
        cgmlp = self._create_conv_gating_mlp(prefix_name, ff_module1, i)

        # merge two branches
        merge_module = self._create_merge_module(
            prefix_name, source=ff_module1, global_extracter=mhsa, local_extracter=cgmlp, layer_index=i
        )

        ff_module2 = self._create_ff_module(prefix_name, 2, merge_module, i)

        res = ff_module2
        if self.block_final_norm:
            res = self.network.add_layer_norm_layer("{}_ln".format(prefix_name), res)
        if self.subsample:
            assert 0 <= i - 1 < len(self.subsample)
            subsample_factor = self.subsample_list[i - 1]
            if subsample_factor > 1:
                res = self.network.add_pool_layer(res + "_pool{}".format(i), res, pool_size=(subsample_factor,))
        res = self.network.add_copy_layer(prefix_name, res)
        return res
