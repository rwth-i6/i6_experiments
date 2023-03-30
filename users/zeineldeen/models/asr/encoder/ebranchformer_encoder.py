from i6_experiments.users.zeineldeen.modules.network import ReturnnNetwork


class EBranchformerEncoder:
    """
    Implement E-branchformer Encoder Architecture
    * Ref: https://arxiv.org/pdf/2210.00077.pdf
    """

    def __init__(self, input='data', input_layer='conv-6', input_layer_conv_act='relu', num_blocks=16,
                 conv_kernel_size=32, specaug=True, pos_enc='rel', activation='swish', ff_dim=512,
                 ff_bias=True, ctc_loss_scale=None, dropout=0.1, att_dropout=0.1, enc_key_dim=256, att_num_heads=4,
                 target='bpe', l2=0.0, lstm_dropout=0.1, rec_weight_dropout=0., with_ctc=False, native_ctc=False,
                 ctc_dropout=0., ctc_l2=0., ctc_opts=None, subsample=None, start_conv_init=None, conv_module_init=None,
                 mhsa_init=None, mhsa_out_init=None, ff_init=None, rel_pos_clipping=16, dropout_in=0.1,
                 batch_norm_opts=None, self_att_l2=0.0, sandwich_conv=False,
                 add_to_prefix_name=None, output_layer_name='encoder', rezero=False):
        """
        :param str input: input layer name
        :param str input_layer: type of input layer which does subsampling
        :param int num_blocks: number of Conformer blocks
        :param int conv_kernel_size: kernel size for conv layers in Convolution module
        :param bool|None specaug: If true, then SpecAug is appliedi wi
        :param str|None activation: activation used to sandwich modules
        :param bool final_norm: if True, apply layer norm to the output of the encoder
        :param int|None ff_dim: dimension of the first linear layer in FF module
        :param str|None ff_init: FF layers initialization
        :param bool|None ff_bias: If true, then bias is used for the FF layers
        :param float embed_dropout: dropout applied to the source embedding
        :param float dropout: general dropout
        :param float att_dropout: dropout applied to attention weights
        :param int enc_key_dim: encoder key dimension, also denoted as d_model, or d_key
        :param int att_num_heads: the number of attention heads
        :param str target: target labels key name
        :param float l2: add L2 regularization for trainable weights parameters
        :param float lstm_dropout: dropout applied to the input of the LSTMs in case they are used
        :param float rec_weight_dropout: dropout applied to the hidden-to-hidden weight matrices of the LSTM in case used
        :param bool with_ctc: if true, CTC loss is used
        :param bool native_ctc: if true, use returnn native ctc implementation instead of TF implementation
        :param float ctc_dropout: dropout applied on input to ctc
        :param float ctc_l2: L2 applied to the weight matrix of CTC softmax
        :param dict[str] ctc_opts: options for CTC
        :param bool rezero: rezero initialization, ref: https://arxiv.org/abs/2003.04887
        """

        self.input = input
        self.input_layer = input_layer
        self.input_layer_conv_act = input_layer_conv_act

        self.num_blocks = num_blocks
        self.conv_kernel_size = conv_kernel_size

        self.pos_enc = pos_enc
        self.rel_pos_clipping = rel_pos_clipping

        self.ff_bias = ff_bias

        self.specaug = specaug

        self.activation = activation

        self.dropout = dropout
        self.att_dropout = att_dropout
        self.lstm_dropout = lstm_dropout

        self.dropout_in = dropout_in

        # key and value dimensions are the same
        self.enc_key_dim = enc_key_dim
        self.enc_value_dim = enc_key_dim
        self.att_num_heads = att_num_heads
        self.enc_key_per_head_dim = enc_key_dim // att_num_heads
        self.enc_val_per_head_dim = enc_key_dim // att_num_heads

        self.ff_dim = ff_dim
        if self.ff_dim is None:
            self.ff_dim = 2 * self.enc_key_dim

        self.target = target

        self.l2 = l2
        self.self_att_l2 = self_att_l2
        self.rec_weight_dropout = rec_weight_dropout

        if batch_norm_opts is None:
            batch_norm_opts = {}

        bn_momentum = batch_norm_opts.pop('momentum', 0.1)
        bn_eps = batch_norm_opts.pop('epsilon', 1e-3)
        bn_update_sample_only_in_train = batch_norm_opts.pop('update_sample_only_in_training', True)
        bn_delay_sample_update = batch_norm_opts.pop('delay_sample_update', True)
        self.batch_norm_opts = {
            'momentum': bn_momentum,
            'epsilon': bn_eps,
            'update_sample_only_in_training': bn_update_sample_only_in_train,
            'delay_sample_update': bn_delay_sample_update,
        }
        self.batch_norm_opts.update(**batch_norm_opts)

        self.with_ctc = with_ctc
        self.native_ctc = native_ctc
        self.ctc_dropout = ctc_dropout
        self.ctc_loss_scale = ctc_loss_scale
        self.ctc_l2 = ctc_l2
        self.ctc_opts = ctc_opts
        if not self.ctc_opts:
            self.ctc_opts = {}

        self.start_conv_init = start_conv_init
        self.conv_module_init = conv_module_init
        self.mhsa_init = mhsa_init
        self.mhsa_out_init = mhsa_out_init
        self.ff_init = ff_init

        self.sandwich_conv = sandwich_conv

        # add maxpooling layers
        self.subsample = subsample
        self.subsample_list = [1] * num_blocks
        if subsample:
            for idx, s in enumerate(map(int, subsample.split('_')[:num_blocks])):
                self.subsample_list[idx] = s

        self.network = ReturnnNetwork()

        self.add_to_prefix_name = add_to_prefix_name
        self.output_layer_name = output_layer_name

        self.rezero = rezero


    def _create_ff_module(self, prefix_name, i, source, block_scale_var):
        """
        Add Feed Forward Module:
        LN -> FFN -> Swish -> Dropout -> FFN -> Dropout
        :param str prefix_name: some prefix name
        :param int i: FF module index
        :param str source: name of source layer
        :return: last layer name of this module
        :rtype: str
        """
        prefix_name = prefix_name + '_ffmod_{}'.format(i)

        ln = self.network.add_layer_norm_layer('{}_ln'.format(prefix_name), source)

        ff1 = self.network.add_linear_layer(
            '{}_ff1'.format(prefix_name), ln, n_out=self.ff_dim, l2=self.l2, forward_weights_init=self.ff_init,
            with_bias=self.ff_bias)

        swish_act = self.network.add_activation_layer('{}_swish'.format(prefix_name), ff1, activation='swish')

        drop1 = self.network.add_dropout_layer('{}_drop1'.format(prefix_name), swish_act, dropout=self.dropout)

        ff2 = self.network.add_linear_layer(
            '{}_ff2'.format(prefix_name), drop1, n_out=self.enc_key_dim, l2=self.l2, forward_weights_init=self.ff_init,
            with_bias=self.ff_bias)

        drop2 = self.network.add_dropout_layer('{}_drop2'.format(prefix_name), ff2, dropout=self.dropout)

        if self.rezero:
            drop2 = self.network.add_eval_layer('{}_scaled_dropout'.format(prefix_name), [block_scale_var, drop2], eval='source(0) * source(1)')

        half_step_ff = self.network.add_eval_layer('{}_half_step'.format(prefix_name), drop2, eval='0.5 * source(0)')
        
        ff_module_res = self.network.add_combine_layer(
            '{}_res'.format(prefix_name), kind='add', source=[half_step_ff, source], n_out=self.enc_key_dim)

        return ff_module_res


    def _create_global_extractor(self, prefix_name, source):
        prefix_name = '{}_global_extractor'.format(prefix_name)

        ln = self.network.add_layer_norm_layer('{}_ln'.format(prefix_name), source)
        
        ln_rel_pos_enc = self.network.add_relative_pos_encoding_layer(
            '{}_ln_rel_pos_enc'.format(prefix_name), ln, n_out=self.enc_key_per_head_dim, forward_weights_init=self.ff_init)
        
        mhsa = self.network.add_self_att_layer(
            '{}'.format(prefix_name), ln, n_out=self.enc_value_dim, num_heads=self.att_num_heads,
            total_key_dim=self.enc_key_dim, att_dropout=self.att_dropout, forward_weights_init=self.ff_init,
            key_shift=ln_rel_pos_enc if ln_rel_pos_enc is not None else None)
        
        mhsa_linear = self.network.add_linear_layer(
            '{}_linear'.format(prefix_name), mhsa, n_out=self.enc_key_dim, l2=self.l2, forward_weights_init=self.ff_init,
            with_bias=False)
        
        dropout = self.network.add_dropout_layer('{}_dropout'.format(prefix_name), mhsa_linear, dropout=self.dropout)

        return dropout
    

    def _create_local_extractor(self, prefix_name, source):
        prefix_name = '{}_local_extractor'.format(prefix_name)

        ln = self.network.add_layer_norm_layer('{}_ln'.format(prefix_name), source)

        ff1 = self.network.add_linear_layer(
            '{}_ff_1'.format(prefix_name), ln, n_out=6*self.enc_key_dim, l2=self.l2, forward_weights_init=self.ff_init,
            with_bias=False)
        
        gelu_act = self.network.add_activation_layer('{}_gelu'.format(prefix_name), ff1, activation='gelu')

        br_part_A = self.network.add_slice_layer('{}_branch_a'.format(prefix_name), gelu_act, 'F', slice_start=0, slice_end =self.enc_key_dim*3)

        br_part_B = self.network.add_slice_layer('{}_branch_b'.format(prefix_name), gelu_act, 'F', slice_start =self.enc_key_dim*3)

        br_part_B_ln = self.network.add_layer_norm_layer('{}_branch_b_ln'.format(prefix_name), br_part_B)

        br_part_B_dpt_conv = self.network.add_conv_layer(
            '{}_branch_b_dpt_conv'.format(prefix_name), br_part_B_ln, n_out=self.enc_key_dim*3,
            filter_size=(self.conv_kernel_size,), groups=self.enc_key_dim*3, l2=self.l2)
        
        br_merge = self.network.add_eval_layer('{}_branch_merge'.format(prefix_name), [br_part_A,br_part_B_dpt_conv], 'source(0)*source(1)')

        dropout = self.network.add_dropout_layer('{}_dropout'.format(prefix_name), br_merge, dropout=self.dropout)

        br_merge_ff = self.network.add_linear_layer(
            '{}_ff_2'.format(prefix_name), dropout, n_out=self.enc_key_dim, l2=self.l2, forward_weights_init=self.ff_init,
            with_bias=False)
        
        return br_merge_ff
    

    def _create_merge_mod(self, prefix_name, source, block_scale_var):
        prefix_name = '{}_merge_mod'.format(prefix_name)

        glb_ext = self._create_global_extractor(prefix_name, source)

        lcl_ext = self._create_local_extractor(prefix_name, source)

        glb_lcl_merge = self.network.add_copy_layer('{}_global_local_merge'.format(prefix_name), [glb_ext, lcl_ext])

        dpt_conv = self.network.add_conv_layer(
            '{}_dpt_conv'.format(prefix_name), glb_lcl_merge, n_out= 2*self.enc_key_dim,
            filter_size=(self.conv_kernel_size,), groups= 2*self.enc_key_dim, l2=self.l2)
        
        dpt_conv_res = self.network.add_combine_layer(
            '{}_dpt_conv_res'.format(prefix_name), kind='add', source=[glb_lcl_merge, dpt_conv], n_out=2*self.enc_key_dim)
        
        ff = self.network.add_linear_layer(
            '{}_ff'.format(prefix_name), dpt_conv_res, n_out=self.enc_key_dim, l2=self.l2, forward_weights_init=self.ff_init,
            with_bias=False)
        
        dropout = self.network.add_dropout_layer('{}_dropout'.format(prefix_name), ff, dropout=self.dropout)

        if self.rezero:
            dropout = self.network.add_eval_layer('{}_scaled_dropout'.format(prefix_name), [block_scale_var, dropout], eval='source(0) * source(1)')

        merge_mod_res = self.network.add_combine_layer(
          '{}_res'.format(prefix_name), kind='add', source=[source, dropout], n_out=self.enc_key_dim)
        
        return merge_mod_res


    def _create_e_branchformer_block(self, i, source):
        prefix_name = 'ebranchformer_block_%02i' % i

        if self.rezero:
            self.network["mod_%02i_var" % i] = {
                "class": "variable", "init":1e-8, "trainable":True, "add_batch_axis":True, "shape":(1,)
            }

        ff_module1 = self._create_ff_module(prefix_name, 1, source, "mod_%02i_var" % i)

        merge_module = self._create_merge_mod(prefix_name, ff_module1, "mod_%02i_var" % i)

        ff_module2 = self._create_ff_module(prefix_name, 2, merge_module, "mod_%02i_var" % i)

        block_ln = self.network.add_layer_norm_layer('{}_ln'.format(prefix_name), ff_module2)

        block_ln = self.network.add_copy_layer(prefix_name, block_ln)
            
        return block_ln


    def _create_all_network_parts(self):
        """
            ConvSubsampling/LSTM -> Linear -> Dropout -> [Conformer Blocks] x N
        """

        data = self.input
        if self.specaug:
            data = self.network.add_eval_layer(
                'source', data,
                eval="self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)")

        subsampled_input = None
        if self.input_layer is None:
            subsampled_input = data
        elif 'lstm' in self.input_layer:
            sample_factor = int(self.input_layer.split('-')[1])
            pool_sizes = None
            if sample_factor == 2:
                pool_sizes = [2, 1]
            elif sample_factor == 4:
                pool_sizes = [2, 2]
            elif sample_factor == 6:
                pool_sizes = [3, 2]
            # add 2 LSTM layers with max pooling to subsample and encode positional information
            subsampled_input = self.network.add_lstm_layers(
                data, num_layers=2, lstm_dim=self.enc_key_dim, dropout=self.lstm_dropout, bidirectional=True,
                rec_weight_dropout=self.rec_weight_dropout, l2=self.l2, pool_sizes=pool_sizes)
        elif self.input_layer == 'conv-4':
            # conv-layer-1: 3x3x32 followed by max pool layer on feature axis (1, 2)
            # conv-layer-2: 3x3x64 with striding (2, 1) on time axis
            # conv-layer-3: 3x3x64 with striding (2, 1) on time axis

            # TODO: make this more generic

            conv_input = self.network.add_conv_block(
                'conv_out', data, hwpc_sizes=[((3, 3), (1, 2), 32)],
                l2=self.l2, activation=self.input_layer_conv_act, init=self.start_conv_init, merge_out=False)

            subsampled_input = self.network.add_conv_block(
                'conv_merged', conv_input, hwpc_sizes=[((3, 3), (2, 1), 64), ((3, 3), (2, 1), 64)],
                l2=self.l2, activation=self.input_layer_conv_act, init=self.start_conv_init, use_striding=True,
                split_input=False, prefix_name='subsample_')
        elif self.input_layer == 'conv-6':
            conv_input = self.network.add_conv_block(
                'conv_out', data, hwpc_sizes=[((3, 3), (1, 2), 32)],
                l2=self.l2, activation=self.input_layer_conv_act, init=self.start_conv_init, merge_out=False)

            subsampled_input = self.network.add_conv_block(
                'conv_merged', conv_input, hwpc_sizes=[((3, 3), (3, 1), 64), ((3, 3), (2, 1), 64)],
                l2=self.l2, activation=self.input_layer_conv_act, init=self.start_conv_init, use_striding=True,
                split_input=False, prefix_name='subsample_')

        assert subsampled_input is not None

        source_linear = self.network.add_linear_layer(
            'source_linear', subsampled_input, n_out=self.enc_key_dim, l2=self.l2, forward_weights_init=self.ff_init,
            with_bias=False)

        if self.dropout_in:
            source_linear = self.network.add_dropout_layer('source_dropout', source_linear, dropout=self.dropout_in)

        conformer_block_src = source_linear
        for i in range(1, self.num_blocks + 1):
            conformer_block_src = self._create_e_branchformer_block(i, conformer_block_src)

        encoder = self.network.add_copy_layer(self.output_layer_name, conformer_block_src)

        if self.with_ctc:
            default_ctc_loss_opts = {'beam_width': 1}
            if self.native_ctc:
                default_ctc_loss_opts['use_native'] = True
            else:
                self.ctc_opts.update({"ignore_longer_outputs_than_inputs": True})  # always enable
            if self.ctc_opts:
                default_ctc_loss_opts['ctc_opts'] = self.ctc_opts
            self.network.add_softmax_layer(
                'ctc', encoder, l2=self.ctc_l2, target=self.target, loss='ctc', dropout=self.ctc_dropout,
                loss_opts=default_ctc_loss_opts, loss_scale=self.ctc_loss_scale)

        return encoder


    def _create_e_branchformer_blocks(self, input):
        conformer_block_src = input
        for i in range(1, self.num_blocks + 1):
            conformer_block_src = self._create_e_branchformer_block(i, conformer_block_src)
        encoder = self.network.add_copy_layer(self.output_layer_name, conformer_block_src)
        return encoder


    def create_network(self):
        return self._create_all_network_parts()


        
