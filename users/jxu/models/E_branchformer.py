__all__ = ['e_branchformer_baseline']

"""
add groups in depth-wise conv layer
"""

from .returnn_network import _NetworkMakerHelper

def e_branchformer_baseline(**kwargs):
  e_branchformer_encoder = EBranchformer(**kwargs)
  e_branchformer_encoder.create_network()
  return e_branchformer_encoder.network.get_net()


class EBranchformer:
    """
    Implement E-branchformer Encoder Architecture
    * Ref: https://arxiv.org/pdf/2210.00077.pdf
    """

    def __init__(self, source='data', use_specaugment=None, num_blocks=6,
                 ff_init=None, ff_bias=True, ff_dim=256, conv_kernel_size=32,
                 enc_key_dim=256, cgMLP_dim=2304, att_dropout=0.1, att_num_heads=4, 
                 reuse_upsample_params=True, iterated_loss_layers=None, iterated_loss_scale={'4': 0.3, '8': 0.3, '12': 0.3},
                 tconv_act='swish', tconv_filter_size=3, tconv_l2 = 0.01,
                 l2=None, dropout=0.1, embed_dropout=0.1, ce_loss_ops={},
                 rezero=False, multi_layer_repr=False):
        """
        :param str source: input layer name
        :param int num_blocks: number of Conformer blocks
        :param str|None ff_init: FF layers initialization
        :param int|None ff_dim: dimension of the first linear layer in FF module
        :param str|None ff_init: FF layers initialization
        :param bool|None ff_bias: If true, then bias is used for the FF layers
        :param int conv_kernel_size: kernel size for conv layers in Convolution module
        :param int enc_key_dim: encoder key dimension, also denoted as d_model, or d_key
        :param int cgMLP_dim: the dimension for the cgMLP
        :param float dropout: general dropout
        :param float embed_dropout: dropout applied to the source embedding
        :param float att_dropout: dropout applied to attention weights
        :param float l2: add L2 regularization for trainable weights parameters
        :param int att_num_heads: the number of attention heads
        :param bool reuse_upsample_params: If ture, then tying the transposed convolution params
        :param list iterated_loss_layers: the list of intermediate layer indices that use auxiliary loss
        :param dict[str] iterated_loss_scale: define the loss scale use for the iterated loss
        :param str|None tconv_act: activation used for transposed convolution in upsampling
        :param str|None tconv_filter_size: filter sizer of transposed convolution in upsampling
        :param dict[str] ce_loss_ops: options for cross entropy loss
        :param bool rezero: rezero initialization, ref: https://arxiv.org/abs/2003.04887
        :param bool multi_layer_repr: aggregating all conformer blocks for output, ref: https://aclanthology.org/C18-1255/

        """
        self.source = source
        self.use_specaugment = use_specaugment
        self.num_blocks = num_blocks
        self.conv_kernel_size = conv_kernel_size

        self.ff_init = ff_init
        self.ff_bias = ff_bias
        self.ff_dim = ff_dim

        self.att_dropout = att_dropout
        self.enc_key_dim = enc_key_dim
        self.enc_value_dim = enc_key_dim
        self.cgMLP_dim = cgMLP_dim
        self.att_num_heads = att_num_heads
        self.enc_key_per_head_dim = enc_key_dim // att_num_heads
        self.enc_val_per_head_dim = enc_key_dim // att_num_heads

        self.reuse_upsample_params = reuse_upsample_params
        self.iterated_loss_layers = iterated_loss_layers
        self.iterated_loss_scale = iterated_loss_scale

        self.tconv_act = tconv_act
        self.tconv_filter_size = tconv_filter_size
        self.tconv_l2 = tconv_l2

        self.dropout = dropout
        self.embed_dropout = embed_dropout
        self.l2 = l2
        self.ce_loss_ops = ce_loss_ops

        self.rezero = rezero
        self.multi_layer_repr = multi_layer_repr

        self.network = _NetworkMakerHelper()


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
        prefix_name = '{}_glb_ext'.format(prefix_name)

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
        prefix_name = '{}_lcl_ext'.format(prefix_name)

        ln = self.network.add_layer_norm_layer('{}_ln'.format(prefix_name), source)

        ff1 = self.network.add_linear_layer(
            '{}_ff_1'.format(prefix_name), ln, n_out=self.cgMLP_dim, l2=self.l2, forward_weights_init=self.ff_init,
            with_bias=False)
        
        gelu_act = self.network.add_activation_layer('{}_gelu'.format(prefix_name), ff1, activation='gelu')

        br_part_A = self.network.add_slice_layer('{}_br_part_A'.format(prefix_name), gelu_act, 'F', slice_start=0, slice_end =self.cgMLP_dim//2)

        br_part_B = self.network.add_slice_layer('{}_br_part_B'.format(prefix_name), gelu_act, 'F', slice_start =self.cgMLP_dim//2)

        br_part_B_ln = self.network.add_layer_norm_layer('{}_br_part_B_ln'.format(prefix_name), br_part_B)

        br_part_B_dpt_conv = self.network.add_conv_layer(
            '{}_br_part_B_dpt_conv'.format(prefix_name), br_part_B_ln, n_out=self.cgMLP_dim//2,
            filter_size=(self.conv_kernel_size,), groups=self.cgMLP_dim//2, l2=self.l2)
        
        br_merge = self.network.add_eval_layer('{}_br_merge'.format(prefix_name), [br_part_A,br_part_B_dpt_conv], 'source(0)*source(1)')

        br_merge_ln = self.network.add_linear_layer(
            '{}_ff_2'.format(prefix_name), br_merge, n_out=self.enc_key_dim, l2=self.l2, forward_weights_init=self.ff_init,
            with_bias=False)

        dropout = self.network.add_dropout_layer('{}_dropout'.format(prefix_name), br_merge_ln, dropout=self.dropout)
        
        return dropout
    

    def _create_merge_mod(self, prefix_name, source, block_scale_var):
        prefix_name = '{}_merge_mod'.format(prefix_name)

        glb_ext = self._create_global_extractor(prefix_name, source)

        lcl_ext = self._create_local_extractor(prefix_name, source)

        glb_lcl_merge = self.network.add_copy_layer('{}_glb_lcl_merge'.format(prefix_name), [glb_ext, lcl_ext])

        dpt_conv = self.network.add_conv_layer(
            '{}_br_part_B_dpt_conv'.format(prefix_name), glb_lcl_merge, n_out= 2*self.enc_key_dim,
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
        prefix_name = 'conformer_block_%02i' % i

        if self.rezero:
            self.network["mod_%02i_var"%i] = {"class": "variable", "init":1e-8, "trainable":True, "add_batch_axis":True, "shape":(1,)}

        ff_module1 = self._create_ff_module(prefix_name, 1, source, "mod_%02i_var"%i)

        merge_module = self._create_merge_mod(prefix_name, ff_module1, "mod_%02i_var"%i)

        ff_module2 = self._create_ff_module(prefix_name, 2, merge_module, "mod_%02i_var"%i)

        block_ln = self.network.add_layer_norm_layer('{}_ln'.format(prefix_name), ff_module2)     

        if self.multi_layer_repr:
            self.network["%s_reinterpret"%block_ln] = {'class': 'reinterpret_data', 'enforce_batch_major':True, 'from': block_ln, 'set_axes': {'T':1, 'F': 2}} 
            block_ln = self.network.add_copy_layer(prefix_name, "%s_reinterpret"%block_ln)
            
        return block_ln


    def create_network(self):
        data = self.source

        if self.use_specaugment:
            data = self.network.add_eval_layer('source', data,eval="self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)")
        
        source_split = self.network.add_split_dim_layer('source0', data)

        self.network["c1"] = {"class" : "conv", "n_out" : 32, "filter_size": (3,3), "padding": "same", "with_bias": True, "from": source_split}
        self.network["y1"] = {"class": "activation", "activation": self.tconv_act, "batch_norm": False, "from": "c1"}
        self.network["p1"] = {"class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "y1"}
        self.network["c3"] = {"class" : "conv", "n_out" : 64, "filter_size": (3,3), "padding": "same",  "with_bias": True,"from" : "p1" }
        self.network["y3"] = {"class": "activation", "activation": self.tconv_act, "batch_norm": False, "from": "c3"}
        self.network["c4"] = {"class" : "conv", "n_out" : 64, "filter_size": (3,3), "padding": "same",  "with_bias": True, "from" : "y3" }
        self.network["y4"] = {"class": "activation", "activation": self.tconv_act, "batch_norm": False, "from": "c4"}
        self.network["p2"] = {"class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "y4"}
        self.network["c2"] = {"class" : "conv", "n_out" : 32, "filter_size": (self.tconv_filter_size,self.tconv_filter_size), "strides": (self.tconv_filter_size, 1), "padding": "same","with_bias": True,"from" : "y4" }  # downsample time
        self.network["y2"] = {"class": "activation", "activation": self.tconv_act, "batch_norm": False, "from": "c2"}
        self.network["vgg_conv_merged"] = {"class": "merge_dims", "from": "y2", "axes": "static"}
        
        source_linear = self.network.add_linear_layer(
            'source_linear', "vgg_conv_merged", n_out=self.enc_key_dim, l2=self.l2, forward_weights_init=self.ff_init,
            with_bias=False)

        source_linear_ln = self.network.add_layer_norm_layer('source_linear_ln', source_linear)

        source_dropout = self.network.add_dropout_layer('source_dropout', source_linear_ln, dropout=self.embed_dropout)

        encoder_block_src = source_dropout

        for i in range(1, self.num_blocks + 1):
            encoder_block_src = self._create_e_branchformer_block(i, encoder_block_src)

            if self.iterated_loss_layers != None and i in self.iterated_loss_layers:
                self.network["transposedconv_%s"%i] = { "class" :"transposed_conv", "n_out" : 512, "filter_size" : [self.tconv_filter_size], "strides": [self.tconv_filter_size],
                                                        "activation": self.tconv_act, "dropout": 0.1, "L2": self.tconv_l2, "from" : encoder_block_src}
                if self.reuse_upsample_params:
                    self.network["transposedconv_%s"%i]["reuse_params"] = "transposedconv"          

                    self.network["masked_tconv_%s"%i] = {"class": "reinterpret_data", "from":"transposedconv_%s"%i, "size_base": "data:classes"}
                    
                    aux_loss_layer = self.network.add_copy_layer('aux_output_block_%s'%i, "masked_tconv_%s"%i)
                    
                    aux_MLP = self.network.add_linear_layer('aux_MLP_block_%s'%i, aux_loss_layer, n_out=self.enc_key_dim, l2=self.l2, forward_weights_init=self.ff_init,with_bias=self.ff_bias)
                    
                    self.network['aux_output_block_%s_ce'%i] = {'class': "softmax", 'dropout': 0.0, 'from': aux_MLP, 'loss_opts':self.ce_loss_ops, 'loss':'ce', 'target': 'classes','loss_scale': self.iterated_loss_scale[str(i)]}
        
        encoder = self.network.add_copy_layer('encoder', encoder_block_src)

        if self.multi_layer_repr:
            all_blocks = ["conformer_block_%02d"%i for i in range(1, self.num_blocks+1)]
            self.network.add_copy_layer("blocks_concat", all_blocks)
            self.network["blocks_concat_ln"] = {'class': 'layer_norm', 'from': 'blocks_concat'}
            self.network.add_linear_layer('blocks_concat_ln_ff', 'blocks_concat_ln', n_out=self.enc_key_dim, l2=self.l2, forward_weights_init=self.ff_init,with_bias=self.ff_bias)
            # self.network["blocks_concat_reinterpret"] = {'class': 'reinterpret_data', 'enforce_batch_major':True, 'from': 'blocks_concat_ln_ff', 'set_axes': {'F': 2}}
            encoder = "blocks_concat_ln_ff"
        
        self.network["transposedconv"] = { "class" :"transposed_conv", "n_out" : 512, "filter_size" : [self.tconv_filter_size], "strides": [self.tconv_filter_size], 
                                        "activation": self.tconv_act, "dropout": 0.1, "L2": self.tconv_l2, "from" :  encoder}
        
        self.network["masked_tconv"] = {"class": "reinterpret_data", "from": "transposedconv", "size_base": "data:classes"}

        output_MLP = self.network.add_linear_layer('MLP_output', "masked_tconv", n_out=self.enc_key_dim, l2=self.l2, forward_weights_init=self.ff_init,with_bias=self.ff_bias)

        self.network['output'] = {
        'class': "softmax", 'dropout': 0.0, 'from': output_MLP,
        'loss': "ce", 'loss_opts': self.ce_loss_ops, 'target': 'classes', 'loss_scale': 1,
        }        



        