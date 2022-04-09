print = lambda x : x # Don't print anything! Other wise the old returnn setups fail!
import textwrap

ext_lm_scale = 0.1
ext_am_scale = 0.1

def local_fusion(source, **kwargs):
    # see /u/zeyer/setups/switchboard/2020-06-09--e2e-multi-gpu/localfusion.conv2l.specaug4a.wdrop03.adrop01.l2a_0001.ctc.devtrain.sgd.lr1e_3.lrwa.lrt_0005.mgpu4.htd100
    """
    :param tf.Tensor am_score: (batch,vocab) in +log space
    :param tf.Tensor lm_score: (batch,vocab) in +log space
    """
    import tensorflow as tf
    from TFUtil import safe_log
    import TFCompat
    am_score = source(0)
    lm_score = source(1)
    am_data = source(0, as_data=True)
    out = ext_am_scale * safe_log(am_score) + ext_lm_scale * safe_log(lm_score)
    return out - TFCompat.v1.reduce_logsumexp(out, axis=am_data.feature_dim_axis, keepdims=True)

class AttEncDecoderModel:
    network = None

    def __init__(self, 
                # Encoder:
                use_conv2l = True,
                specaug4a = True,
                num_lstm_layer = 8,
                lstm_dim = 1024,
                l2 = None,
                pooling = None,
                batch_norm = False,
                bn_until_condition = None,
                skip_con_at_layer = -1, # -1, no skipcon, int, all skip, list -> specific skip
                dropout = 0.0,
                enc_key_dim = 1024,

                # Decoder:
                att_num_heads = 1,
                add_lm = False,
                
                # General:
                is_training = True,
                global_train_epoch = 0,
                target = "classes",

                ext_lm_scale = 0.1,
                ext_am_scale = 0.1,

                net_summary_to_file = None
                ): 
        # initalize all things

        self.use_conv2l = use_conv2l
        self.specaug4a = specaug4a
        self.num_lstm_layer = num_lstm_layer

        self.lstm_dim = lstm_dim
        self.l2 = l2
        self.dropout = dropout
        self.enc_key_dim = enc_key_dim
        self.add_lm = add_lm

        self.att_num_heads = att_num_heads

        self.batch_norm = batch_norm
        self.bn_until_condition = bn_until_condition # TODO: repace with persormance measure

        self.net_summary_to_file = net_summary_to_file

        self.target = target

        self.ext_lm_scale = ext_lm_scale # has to be passed since it can be tuned in the config
        globals()["ext_lm_scale"] = self.ext_lm_scale # TODO: not needed any more

        self.ext_am_scale = ext_am_scale 
        globals()["ext_am_scale"] = self.ext_am_scale

        if self.bn_until_condition is not None:
            print("WARNING: conditional batch norm works only with custom training pipeline \n please use update_network = _model.custom_pipeline in config")

        # This stuff should not bee needed when using sisyphus
        self.is_training = is_training
        self.global_train_epoch = global_train_epoch

        assert self.global_train_epoch == 0, "ERR: loading state not jet implemented"

        self.enc_value_dim = self.enc_key_dim * 2
        self.enc_key_per_head_dim = self.enc_key_dim // self.att_num_heads
        self.enc_value_per_head_dim = self.enc_value_dim // self.att_num_heads

        assert lstm_dim * 2 == self.enc_value_dim, "Dim mismatch, maybe forgot subsapling?"

        if pooling is None:
            pooling = [None] * self.num_lstm_layer
        elif type(pooling) == list:
            if len(pooling) < self.num_lstm_layer:
                pooling.append([None]*(self.num_lstm_layers - len(pooling)))

        self.pooling = pooling
        
        if skip_con_at_layer == -1:
            self.skip_con_at_layer = [False] * self.num_lstm_layer
        else:
            assert False, "skip con not useable with this build jet"

        

        self.network = _NetworkMakerHelper()

    
    def _add_lstm_blocks(self, source):
        # Alls all lstm blocks
        _from = source

        for i in range(self.num_lstm_layer):
            if self.skip_con_at_layer[i]:
                # add skip con
                pass
            _from = self._add_lstm_fw_bw_block(_from, i)

            if self.pooling[i] != None:
                _from = self.network.update_net({'lstm%i_pool' % i: {'class': 'pool', 'mode': 'max', 'padding': 'same', 'pool_size': self.pooling[i], 'from': _from, 'trainable': False}})

            if self.batch_norm:
                kwargs = {} if self.is_training else {"use_sample" : 1} # For inference we should set this the docs say ( default is use_sample = 0 )
                if self.bn_until_condition is not None:
                    # https://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow 
                    # check the condition:
                    if self.bn_until_condition(self.global_train_epoch):
                        _from = self.network.add_batch_norm_layer('batch_norm%i' % i, _from, **kwargs)
                    else:
                        pass # TODO: or should we here return a non trainable batch-norm layer?
                else:
                    _from = self.network.add_batch_norm_layer('batch_norm%i' % i, _from, **kwargs)

        return _from

    def _add_lstm_fw_bw_block(self, source, idx=-1):
        layer = {'lstm%i_fw' % idx: {'class': 'rec', 'unit': 'nativelstm2', 'n_out': self.lstm_dim, 'direction': 1, 'from': source, 'dropout': self.dropout},
                'lstm%i_bw' % idx: {'class': 'rec', 'unit': 'nativelstm2', 'n_out': self.lstm_dim, 'direction': -1, 'from': source, 'dropout': self.dropout}}

        if self.l2 is not None:
            for k in layer:
                layer[k]["L2"] = self.l2
        self.network.update_net(layer)
        return ['lstm%i_fw' % idx, 'lstm%i_bw' % idx] # Wee need both these as input

    def _add_conv_subsampling(self, source):
        return self.network.update_net({
            'source0': {'class': 'split_dims', 'axis': 'F', 'dims': (-1, 1), 'from': source},
            'conv0': {'class': 'conv', 'from': 'source0', 'padding': 'same', 'filter_size': (3, 3), 'n_out': 32, 'activation': None, 'with_bias': True},
            'conv0p': {'class': 'pool', 'mode': 'max', 'padding': 'same', 'pool_size': (1, 2), 'from': 'conv0'},
            'conv1': {'class': 'conv', 'from': 'conv0p', 'padding': 'same', 'filter_size': (3, 3), 'n_out': 32, 'activation': None, 'with_bias': True},
            'conv1p': {'class': 'pool', 'mode': 'max', 'padding': 'same', 'pool_size': (1, 2), 'from': 'conv1'},
            'conv_merged': {'class': 'merge_dims', 'from': 'conv1p', 'axes': 'static'}})

    def _add_encoder_cap(self, source):
        return self.network.update_net({'encoder': {'class': 'copy', 'from': source},
            'enc_ctx': {'class': 'linear', 'activation': None, 'with_bias': True, 'from': ['encoder'], 'n_out': 1024},
            'inv_fertility': {'class': 'linear', 'activation': 'sigmoid', 'with_bias': False, 'from': ['encoder'], 'n_out': 1},
            'enc_value': {'class': 'split_dims', 'axis': 'F', 'dims': (1, self.enc_value_per_head_dim), 'from': ['encoder']}})

    def _add_desigeon_layer(self, source):
        return self.network.update_net({"decision": {
                "class": "decide", "from": ["output"], "loss": "edit_distance", "target": self.target,
                "loss_opts": {}} })
    
    def _add_default_decoder(self, source):
        return self.network.update_net({ "output" : {'class': 'rec', 'from': [], 'unit': { 
            'output': {'class': 'choice', 'target': self.target, 'beam_size': 12, 'from': ['output_prob'], 'initial_output': 0}, 
            'end': {'class': 'compare', 'from': ['output'], 'value': 0},
            'target_embed': {'class': 'linear', 'activation': None, 'with_bias': False, 'from': ['output'], 'n_out': 621, 'initial_output': 0},
            'weight_feedback': {'class': 'linear', 'activation': None, 'with_bias': False, 'from': ['prev:accum_att_weights'], 'n_out': 1024},
            's_transformed': {'class': 'linear', 'activation': None, 'with_bias': False, 'from': ['s'], 'n_out': 1024},
            'energy_in': {'class': 'combine', 'kind': 'add', 'from': ['base:enc_ctx', 'weight_feedback', 's_transformed'], 'n_out': 1024},
            'energy_tanh': {'class': 'activation', 'activation': 'tanh', 'from': ['energy_in']},
            'energy': {'class': 'linear', 'activation': None, 'with_bias': False, 'from': ['energy_tanh'], 'n_out': 1},
            'exp_energy': {'class': 'activation', 'activation': 'exp', 'from': ['energy']},
            'att_weights': {'class': 'softmax_over_spatial', 'from': ['energy']},
            'accum_att_weights': {'class': 'eval', 'from': ['prev:accum_att_weights', 'att_weights', 'base:inv_fertility'], 'eval': 'source(0) + source(1) * source(2) * 0.5', 'out_type': {'dim': 1, 'shape': (None, 1)}},
            'att0': {'class': 'generic_attention', 'weights': 'att_weights', 'base': 'base:enc_value'},
            'att': {'class': 'merge_dims', 'axes': 'except_batch', 'from': ['att0']},
            's': {'class': 'rnn_cell', 'unit': 'LSTMBlock', 'from': ['prev:target_embed', 'prev:att'], 'n_out': 1000},
            'readout_in': {'class': 'linear', 'from': ['s', 'prev:target_embed', 'att'], 'activation': None, 'n_out': 1000}, \
            "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
                "output_prob": {
                    "class": "softmax", "from": ["readout"], "dropout": 0.3,
                    "target": self.target, "loss": "ce", "loss_opts": {"label_smoothing": 0.1}},
            }, "target": self.target, "max_seq_len": "max_len_from('base:encoder')"}})
    
    def _add_language_model(self):
        lm_model_filename = "/work/asr3/irie/experiments/lm/switchboard/2018-01-23--lmbpe-zeyer/data-train/bpe1k_clean_i256_m2048_m2048.sgd_b16_lr0_cl2.newbobabs.d0.2/net-model/network.023"
        self.network.get_net()['output']['unit']['output']['from'] = 'combo_output_log_prob'
        self.network.get_net()['output']['unit']['output']['input_type'] = 'log_prob'

        lm = {"lm_output": { "class": "subnetwork", "from": ["prev:output"], "load_on_init": lm_model_filename, "n_out": 1030,
            "subnetwork": {
            "input": {"class": "linear", "n_out": 256, "activation": "identity"},
            "lstm0": {"class": "rnn_cell", "unit": "LSTMBlock", "dropout": 0.2, "n_out": 2048, "unit_opts": {"forget_bias": 0.0}, "from": ["input"]},
            "lstm1": {"class": "rnn_cell", "unit": "LSTMBlock", "dropout": 0.2, "n_out": 2048, "unit_opts": {"forget_bias": 0.0}, "from": ["lstm0"]},
            "output": {"class": "linear", "from": ["lstm1"], "activation": "identity", "dropout": 0.2, "n_out": 1030}
            }},
            "lm_output_prob" : {"class": "activation", "activation": "softmax", "from": ["lm_output"], "target": self.target},
            'combo_output_log_prob': {"class": "eval", "from": ["output_prob", "lm_output_prob"], "eval": "self.network.get_config().typed_value('local_fusion')(source)" },
            'combo_output_prob': {"class": "eval", "from": ['combo_output_log_prob'], "eval": "tf.exp(source(0))" }}

        self.network.get_net()['output']['unit'].update(lm) # adds all necessary lm code


    def get_construction_code(self):
        return textwrap.dedent(
"""
def custom_construction_algo(idx, net_dict):
    # pool = True
    # For debugging, use: python3 ./crnn/Pretrain.py config... Maybe set repetitions=1 below.
    StartNumLayers = 2
    InitialDimFactor = 0.5
    orig_num_lstm_layers = 0
    while "lstm%i_fw" % orig_num_lstm_layers in net_dict:
        orig_num_lstm_layers += 1
    assert orig_num_lstm_layers >= 2
    orig_red_factor = 1
    if pool:
        for i in range(orig_num_lstm_layers - 1):
            orig_red_factor *= net_dict["lstm%i_pool" % i]["pool_size"][0]
    net_dict["#config"] = {}
    if idx < 4:
        #net_dict["#config"]["batch_size"] = 15000
        net_dict["#config"]["accum_grad_multiple_step"] = 4
    idx = max(idx - 4, 0)  # repeat first
    num_lstm_layers = idx + StartNumLayers  # idx starts at 0. start with N layers
    if num_lstm_layers > orig_num_lstm_layers:
        # Finish. This will also use label-smoothing then.
        return None
    if pool:
        if num_lstm_layers == 2:
            net_dict["lstm0_pool"]["pool_size"] = (orig_red_factor,)
    # Skip to num layers.
    net_dict["encoder"]["from"] = ["lstm%i_fw" % (num_lstm_layers - 1), "lstm%i_bw" % (num_lstm_layers - 1)]
    # Delete non-used lstm layers. This is not explicitly necessary but maybe nicer.
    for i in range(num_lstm_layers, orig_num_lstm_layers):
        del net_dict["lstm%i_fw" % i]
        del net_dict["lstm%i_bw" % i]
        if pool:
            del net_dict["lstm%i_pool" % (i - 1)]
    # Thus we have layers 0 .. (num_lstm_layers - 1).
    layer_idxs = list(range(0, num_lstm_layers))
    layers = ["lstm%i_fw" % i for i in layer_idxs] + ["lstm%i_bw" % i for i in layer_idxs]
    grow_frac = 1.0 - float(orig_num_lstm_layers - num_lstm_layers) / (orig_num_lstm_layers - StartNumLayers)
    dim_frac = InitialDimFactor + (1.0 - InitialDimFactor) * grow_frac
    for layer in layers:
        net_dict[layer]["n_out"] = int(net_dict[layer]["n_out"] * dim_frac)
        if "dropout" in net_dict[layer]:
            net_dict[layer]["dropout"] *= dim_frac
    net_dict["enc_value"]["dims"] = (%att_num_heads, int(%enc_key_pd * dim_frac * 0.5) * 2)
    # Use label smoothing only at the very end.
    net_dict["output"]["unit"]["output_prob"]["loss_opts"]["label_smoothing"] = 0
    return net_dict
""").replace("%att_num_heads", str(self.att_num_heads)).replace("%enc_key_pd", str(self.enc_value_per_head_dim))

    def custom_construction_algo(self, idx, net_dict):
        pool = not (self.pooling[0] == None) # Does the net use pooling at all?

        # For debugging, use: python3 ./crnn/Pretrain.py config... Maybe set repetitions=1 below.
        StartNumLayers = 2
        InitialDimFactor = 0.5
        orig_num_lstm_layers = 0
        while "lstm%i_fw" % orig_num_lstm_layers in net_dict:
            orig_num_lstm_layers += 1
        assert orig_num_lstm_layers >= 2
        orig_red_factor = 1
        if pool:
            for i in range(orig_num_lstm_layers - 1):
                orig_red_factor *= net_dict["lstm%i_pool" % i]["pool_size"][0]
        net_dict["#config"] = {}
        if idx < 4:
            #net_dict["#config"]["batch_size"] = 15000
            net_dict["#config"]["accum_grad_multiple_step"] = 4
        idx = max(idx - 4, 0)  # repeat first
        num_lstm_layers = idx + StartNumLayers  # idx starts at 0. start with N layers
        if num_lstm_layers > orig_num_lstm_layers:
            # Finish. This will also use label-smoothing then.
            return None
        if pool:
            if num_lstm_layers == 2:
                net_dict["lstm0_pool"]["pool_size"] = (orig_red_factor,)
        # Skip to num layers.
        net_dict["encoder"]["from"] = ["lstm%i_fw" % (num_lstm_layers - 1), "lstm%i_bw" % (num_lstm_layers - 1)]
        # Delete non-used lstm layers. This is not explicitly necessary but maybe nicer.
        for i in range(num_lstm_layers, orig_num_lstm_layers):
            del net_dict["lstm%i_fw" % i]
            del net_dict["lstm%i_bw" % i]
            if pool:
                del net_dict["lstm%i_pool" % (i - 1)]
        # Thus we have layers 0 .. (num_lstm_layers - 1).
        layer_idxs = list(range(0, num_lstm_layers))
        layers = ["lstm%i_fw" % i for i in layer_idxs] + ["lstm%i_bw" % i for i in layer_idxs]
        grow_frac = 1.0 - float(orig_num_lstm_layers - num_lstm_layers) / (orig_num_lstm_layers - StartNumLayers)
        dim_frac = InitialDimFactor + (1.0 - InitialDimFactor) * grow_frac
        for layer in layers:
            net_dict[layer]["n_out"] = int(net_dict[layer]["n_out"] * dim_frac)
            if "dropout" in net_dict[layer]:
                net_dict[layer]["dropout"] *= dim_frac
        net_dict["enc_value"]["dims"] = (self.att_num_heads, int(self.enc_value_per_head_dim * dim_frac * 0.5) * 2)
        # Use label smoothing only at the very end.
        net_dict["output"]["unit"]["output_prob"]["loss_opts"]["label_smoothing"] = 0
        return net_dict

    def custom_train_pipeline(self, epoch, **kwargs):
        self.global_epoch = epoch
        return self.create_network()

    def create_network(self):

        self.network.clear_net()
        _from = None 
        # Note(tim): don't need an origin, it's part of the config transform() function -> added in the config

        if self.specaug4a:
            _from = self.network.add_eval_layer("source", _from, "self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)")

        if self.use_conv2l:
            _from = self._add_conv_subsampling(_from)
        
        # now add the lstm_blocks
        _from = self._add_lstm_blocks(_from)

        _from = self._add_encoder_cap(_from)

        _from = self._add_default_decoder(_from)

        _from = self._add_desigeon_layer(_from)

        if self.add_lm:
            self._add_language_model()

        full_net = self.network.get_net()

        if self.net_summary_to_file != None:
            assert type(self.net_summary_to_file) == str, "file path expected"
            with open(self.net_summary_to_file, 'w') as outfile: 
                outfile.write(str(full_net).replace("},", "},\n"))

        return full_net

class _NetworkMakerHelper:
    def __init__(self):
        self._net = {}

    def clear_net(self):
        self._net.clear()

    def get_net(self):
        return self._net

    def update_net(self, _dict):
        for k in _dict:
            self._net[k] = _dict[k]
            last_name = k
        return last_name

    def add_copy_layer(self, name, source, **kwargs):
        self._net[name] = {'class': 'copy'}
        if source is not None:
            self._net[name]["from"] = source
        self._net[name].update(kwargs)
        return name

    def add_eval_layer(self, name, source, eval, **kwargs):
        self._net[name] = {'class': 'eval', 'eval': eval}
        if source is not None:
            self._net[name]["from"] = source
        self._net[name].update(kwargs)
        return name

    def add_batch_norm_layer(self, name, source, **kwargs):
        self._net[name] = {'class': 'batch_norm', 'from': source, **kwargs}
        return name
