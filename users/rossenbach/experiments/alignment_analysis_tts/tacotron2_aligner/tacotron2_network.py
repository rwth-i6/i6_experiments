import copy

from i6_core.returnn.config import ReturnnConfig


def stop_token_target(ramp_length=5):
    """
    builds the custom layer code for the stop token target

    :param ramp_length:
    :return:
    """

    # function declaration, tensorflow and time axis
    string = "\ndef _stop_token_target(data):\n"
    string += "  import tensorflow as tf\n"
    string += "  time_axis = data.get_dynamic_axes()[0]\n"

    # position where the ramp starts is zero, so T - ramp_len - 1
    # get dynamic size is (B,), so we expand to (B, 1) for broadcasting
    stop_offset = ramp_length + 1
    string += "  stop_position = tf.expand_dims(data.get_dynamic_size(time_axis), axis=1) - %i\n" % stop_offset

    # a single ramp of (1, T) as [[0, 1, 2, 3, 4, 5 ..., T]]
    string += "  ramp = tf.expand_dims(tf.range(tf.shape(data.placeholder)[1]), axis=0)\n"

    # the full ramp is now (B, T)
    string += "  full_ramp = tf.tile(ramp, [tf.shape(data.placeholder)[0], 1])\n"

    # now subtract the stop position and limit the values to the range [0, ramp_length]
    string += "  adapted_ramp = tf.minimum(tf.maximum(full_ramp - stop_position, 0), %i)\n" % ramp_length

    # cast to float32, limit to [0, 1] and return
    string += "  return tf.cast(tf.expand_dims(adapted_ramp, 2), dtype=\"float32\") / %i\n\n" % ramp_length

    return string



def stop_token_target_subnet(ramp_length=5):
    subnet = {
        'class': 'subnetwork',
        'from': [],
        'subnetwork': {
            'cast': {'class': 'cast', 'dtype': 'float32', 'from': 'min_max'},
            'div': {'class': 'combine', 'from': ['cast', 'ramp_length'], 'kind': 'truediv'},
            'expand_div': {'axis': 2, 'class': 'expand_dims', 'from': 'div'},
            'ramp_length': {'class': 'constant', 'dtype': 'float32', 'value': ramp_length},
            'length': {'add_time_axis': False, 'class': 'length', 'from': 'base:windowed_data_target'},
            'min_max': {
                'class': 'eval',
                'eval': f"tf.minimum(tf.maximum(source(0), 0), {ramp_length})",
                'from': 'sub_0'},
            'output': {'class': 'copy', 'from': 'expand_div'},
            'ramp_length_incremented': {'class': 'constant', 'value': ramp_length + 1},
            'sub': {'class': 'combine', 'from': ['length', 'ramp_length_incremented'], 'kind': 'sub'},
            'sub_0': { 'class': 'combine', 'from': [ 'time_range', 'sub'], 'kind': 'sub'},
            'time_range': { 'axis': 'T', 'class': 'range_in_axis', 'from': 'base:windowed_data_target', 'keepdims': False}}}
    return subnet


def conv_block(name, index, input_layer, dim, filter_size, L2, dropout, activation="relu", regularization="batch_norm"):
    """
    adds a 1-D convolutional layer with regularization and batch_norm

    :param name:
    :param index:
    :param input_layer:
    :param dim:
    :param L2:
    :param dropout:
    :param regularization
    :return:
    """

    if regularization == "batch_norm":
        net = {
            name + '_conv%i' % index: {'class': 'conv', 'activation': activation, 'from': [input_layer],
                                       'filter_size': (filter_size,),
                                       'padding': 'same', 'n_out': dim, 'L2': L2},
            name + '_batchnorm_cv_%i' % index: {'class': 'batch_norm', 'from': [name + '_conv%i' % index]},
            name + '_conv%i_out' % index: {'class': 'dropout', 'from': [name + '_batchnorm_cv_%i' % index],
                                           'dropout': dropout,
                                           'dropout_noise_shape': {'*': None}},
        }
    elif regularization == "batch_norm_new":
        net = {
            name + '_conv%i' % index: {'class': 'conv', 'activation': activation, 'from': [input_layer],
                                       'filter_size': (filter_size,),
                                       'padding': 'same', 'n_out': dim, 'L2': L2},
            name + '_batchnorm_cv_%i' % index: {'class': 'batch_norm', 'from': [name + '_conv%i' % index],
                                                'momentum': 0.1, 'epsilon': 1e-3,
                                                'update_sample_only_in_training': True, 'delay_sample_update': True},
            name + '_conv%i_out' % index: {'class': 'dropout', 'from': [name + '_batchnorm_cv_%i' % index],
                                           'dropout': dropout,
                                           'dropout_noise_shape': {'*': None}},
        }
    elif regularization == "espnet_batch_norm":
        net = {
            name + '_conv%i' % index: {'class': 'conv', 'activation': None, 'from': [input_layer],
                                       'filter_size': (filter_size,),
                                       'padding': 'same', 'n_out': dim, 'L2': L2},
            name + '_batchnorm_cv_%i' % index: {'class': 'batch_norm', 'from': [name + '_conv%i' % index],
                                                'force_sample': True,
                                                'momentum': 0.1, 'epsilon': 1e-3,
                                                'update_sample_only_in_training': True},
            name + '_%s_%i' % (activation, index): {'class': 'activation', 'activation': activation,
                                                    'from': [name + '_batchnorm_cv_%i' % index]},
            name + '_conv%i_out' % index: {'class': 'dropout', 'from': [name + '_%s_%i' % (activation, index)],
                                           'dropout': dropout,
                                           'dropout_noise_shape': {'*': None}},
        }
    elif regularization == "layer_norm":
        net = {
            name + '_conv%i' % index: {'class': 'conv', 'activation': activation, 'from': [input_layer],
                                       'filter_size': (filter_size,),
                                       'padding': 'same', 'n_out': dim, 'L2': L2},
            name + '_layernorm_cv_%i' % index: {'class': 'layer_norm', 'from': [name + '_conv%i' % index]},
            name + '_conv%i_out' % index: {'class': 'dropout', 'from': [name + '_layernorm_cv_%i' % index],
                                           'dropout': dropout,
                                           'dropout_noise_shape': {'*': None}},
        }
    elif regularization == "pre_act_layer_norm":
        net = {
            name + '_conv%i' % index: {'class': 'conv', 'activation': None, 'from': [input_layer],
                                       'filter_size': (filter_size,),
                                       'padding': 'same', 'n_out': dim, 'L2': L2},
            name + '_layernorm_cv_%i' % index: {'class': 'layer_norm', 'from': [name + '_conv%i' % index]},
            name + '_%s_%i' % (activation, index): {'class': 'activation', 'activation': activation,
                                                    'from': [name + '_layernorm_cv_%i' % index]},
            name + '_conv%i_out' % index: {'class': 'dropout', 'from': [name + '_%s_%i' % (activation, index)],
                                           'dropout': dropout,
                                           'dropout_noise_shape': {'*': None}},
        }
    elif regularization == None:
        net = {
            name + '_conv%i' % index: {'class': 'conv', 'activation': None, 'from': [input_layer],
                                       'filter_size': (filter_size,),
                                       'padding': 'same', 'n_out': dim, 'L2': L2},
            name + '_%s_%i' % (activation, index): {'class': 'activation', 'activation': activation,
                                                    'from': [name + '_conv%i' % index]},
            name + '_conv%i_out' % index: {'class': 'dropout', 'from': [name + '_%s_%i' % (activation, index)],
                                           'dropout': dropout,
                                           'dropout_noise_shape': {'*': None}},
        }
    else:
        assert False

    return net


class Tacotron2NetworkBuilderV2():
    """
    A builder class to construct the network dictionary for the text-to-feature model

    V2: Uses windowing for multi-frame targets, fixes some hacks
    """

    def __init__(self, network_options, stop_token_loss_scale=1.0, postnet_loss_scale=0.25):
        """

        :param dict network_options: the option dict containing the network settings
        """

        self.audio_opt_feature_size = network_options['feature_size']
        self.audio_opt_join_frames = network_options['frame_reduction_factor']

        # encoder options
        self.embedding_dim = network_options["embedding_dim"]
        self.encoder_conv_dims = network_options["encoder_conv_dims"]
        self.encoder_conv_filter_sizes = network_options["encoder_conv_filter_sizes"]
        self.encoder_lstm_dim = network_options["encoder_lstm_dim"]
        self.encoder_lstm_type = network_options["encoder_lstm_type"]

        # optional encoder options
        self.encoder_position_dim = network_options["encoder_position_dim"]
        self.encoder_dropout = network_options["encoder_dropout"]
        self.encoder_regularization = network_options["encoder_regularization"]
        assert self.encoder_regularization in ["batch_norm", "layer_norm", "espnet_batch_norm", "pre_act_layer_norm",
                                               "batch_norm_new", None]
        self.encoder_lstm_dropout = network_options["encoder_lstm_dropout"]
        self.encoder_lstm_dropout_broadcasting = network_options["encoder_lstm_dropout_broadcasting"]

        # attention options
        self.attention_dim = network_options["attention_dim"]
        self.num_location_filters = network_options["num_location_filters"]
        self.location_filter_size = network_options["location_filter_size"]
        assert self.location_filter_size % 2 == 1 and self.location_filter_size >= 3
        self.attention_in_dropout = network_options["attention_in_dropout"]

        self.location_feedback_dropout = network_options["location_feedback_dropout"]

        # decoder options
        self.decoder_dim = network_options["decoder_dim"]
        self.decoder_type = network_options["decoder_type"]
        self.zoneout = network_options.get("zoneout", None)
        self.decoder_dropout = network_options["decoder_dropout"]

        # pre-net options
        self.pre_layer1_dim = network_options['prenet_layer1_dim']
        self.pre_layer2_dim = network_options['prenet_layer2_dim']
        self.pre_layer1_dropout = network_options['prenet_layer1_dropout']
        self.pre_layer2_dropout = network_options['prenet_layer2_dropout']

        # post-net options
        self.post_conv_dims = network_options["post_conv_dims"]
        self.post_conv_filter_sizes = network_options["post_conv_filter_sizes"]
        self.post_dropout = network_options["post_conv_dropout"]

        self.max_decoder_seq_len = network_options["max_decoder_seq_len"]

        # regularization
        self.L2_Norm = network_options["l2_norm"]

        # output options
        self.target_loss = network_options["target_loss"]

        # decoding stufff
        self.decoding_stop_threshold = network_options["decoding_stop_threshold"]
        self.decoding_additional_steps = network_options["decoding_additional_steps"]

        self.stop_token_ramp_length = network_options["stop_token_ramp_length"]

        # speaker stuff
        self.speaker_embedding_size = network_options.get("speaker_embedding_size", 256)

        self.post_net_loss_scale = postnet_loss_scale
        self.stop_token_loss_scale = stop_token_loss_scale

    def _decoder(self):
        """
        create the decoder depending on the selected decoder type

        :return: network dictionary
        :rtype: dict
        """
        if self.decoder_type == "zoneout":
            decoder_conf = {'decoder_1': {'class': 'rnn_cell', 'from': ['pre_net_layer_2_out', 'prev:att0'],
                                          'n_out': self.decoder_dim, 'unit': 'zoneoutlstm',
                                          'unit_opts': {'zoneout_factor_cell': self.zoneout,
                                                        'zoneout_factor_output': self.zoneout}},
                            'decoder_2': {'class': 'rnn_cell', 'from': ['decoder_1'], 'n_out': self.decoder_dim,
                                          'unit': 'zoneoutlstm',
                                          'unit_opts': {'zoneout_factor_cell': self.zoneout,
                                                        'zoneout_factor_output': self.zoneout}}, }
        elif self.decoder_type == "default":
            decoder_conf = {'decoder_1': {'class': 'rnn_cell', 'from': ['pre_net_layer_2_out', 'prev:att0'],
                                          'n_out': self.decoder_dim, 'unit': 'LSTMBlock',
                                          'dropout': self.decoder_dropout},
                            'decoder_2': {'class': 'rnn_cell', 'from': ['decoder_1'], 'n_out': self.decoder_dim,
                                          'unit': 'LSTMBlock', 'dropout': self.decoder_dropout}}
        else:
            assert False, "invalid decoder type %s" % self.decoder_type

        return decoder_conf

    def create_network(self):
        """
        this creates the network for training depending on the set parameters
        :return: network dictionary
        :rtype: dict
        """

        location_filter_half = int((self.location_filter_size - 1) / 2)

        network = {
            # for sparse input, a linear layer is automatically a lookup layer
            'embedding': {'class': 'linear', 'activation': None, 'from': ['data:phon_labels'],
                          'n_out': self.embedding_dim},

            'speaker_label_notime': {'class': 'squeeze', 'axis': 'T',
                                     'from': ["data:speaker_labels"]},
            'speaker_embedding': {'class': 'linear', 'from': ["speaker_label_notime"],
                                  'activation': None, 'n_out': self.speaker_embedding_size},

            # the encoder states are the concatenated LSTM states, a speaker embedding might be added at this point later on
            'encoder': {'class': 'copy', 'from': ['lstm0_fw', 'lstm0_bw', 'speaker_embedding'],
                        'dropout': self.encoder_lstm_dropout},

            # the attention "key" value for the MLP attention is a transformation of the encoder and the positonal encoding
            'enc_ctx': {'activation': None, 'class': 'linear', 'from': ['encoder'],
                        'n_out': self.attention_dim, 'with_bias': True, 'dropout': self.attention_in_dropout,
                        'L2': self.L2_Norm},

            # this is the recurrent block for the decoder
            'decoder': {'cheating': False,
                        'class': 'rec',
                        'from': [],
                        'target': 'windowed_data_target',
                        'max_seq_len': self.max_decoder_seq_len,
                        'unit': {

                            'att_energy_in': {'class': 'combine',
                                              'from': ['base:enc_ctx', 's_transformed',
                                                       'location_feedback_transformed'],
                                              'kind': 'add', 'n_out': self.attention_dim},
                            'att_energy_tanh': {'activation': 'tanh', 'class': 'activation', 'from': ['att_energy_in']},

                            'att_energy': {'activation': None, 'class': 'linear', 'from': ['att_energy_tanh'],
                                           'n_out': 1,
                                           'with_bias': False},
                            'att_weights': {'class': 'softmax_over_spatial', 'from': ['att_energy']},
                            'entropy': {'class': 'eval',
                                        'eval': '-tf.reduce_sum(source(0)*safe_log(source(0)), axis=-1, keepdims=True)',
                                        'from': ['att_weights'], 'loss': 'as_is', 'loss_scale': 1e-4},
                            'accum_att_weights': {'class': 'combine',
                                                  'kind': 'add',
                                                  'from': ['prev:accum_att_weights', 'att_weights'],
                                                  'is_output_layer': True},  # [B, T_enc, 1]
                            'feedback_pad_left': {'class': 'pad', 'axes': 's:0',
                                                  'padding': ((location_filter_half, 0),), 'value': 1,
                                                  'mode': 'constant', 'from': ['prev:accum_att_weights'], },
                            'feedback_pad_right': {'class': 'pad', 'axes': 's:0',
                                                   'padding': ((0, location_filter_half),), 'value': 0,
                                                   'mode': 'constant', 'from': ['feedback_pad_left'], },
                            # [B, T_enc + filter_size, 1]
                            'convolved_att': {'class': 'conv', 'activation': None, 'from': ['feedback_pad_right'],
                                              'filter_size': (self.location_filter_size,), 'padding': 'valid',
                                              'n_out': self.num_location_filters,
                                              'L2': self.L2_Norm},
                            'location_feedback_transformed': {'class': 'linear', 'activation': None, 'with_bias': False,
                                                              'n_out': self.attention_dim, 'from': ['convolved_att'],
                                                              'L2': self.L2_Norm,
                                                              'dropout': self.location_feedback_dropout},
                            'att0': {'base': 'base:encoder', 'class': 'generic_attention', 'weights': 'att_weights'},
                            's_transformed': {'activation': None, 'class': 'linear', 'from': ['decoder_2'],
                                              'n_out': self.attention_dim, 'with_bias': False,
                                              'dropout': self.attention_in_dropout,
                                              'L2': self.L2_Norm},
                            'stop_token': {'class': 'linear', 'activation': None, 'n_out': 1, 'loss': 'bin_ce',
                                           'loss_scale': self.stop_token_loss_scale, 'target': 'stop_token_target',
                                           'from': ['decoder_2', 'att0']},
                            'stop_token_sigmoid': {'class': 'activation', 'activation': 'sigmoid',
                                                   'from': ['stop_token']},
                            'end': {'class': 'compare', 'kind': 'greater', 'from': ['stop_token_sigmoid'],
                                    'value': 0.5},
                            'output': {'activation': None, 'class': 'linear', 'loss': self.target_loss,
                                       'loss_scale': 1.0,
                                       'target': 'windowed_data_target', 'from': ['decoder_2', 'att0'],
                                       'n_out': self.audio_opt_feature_size * self.audio_opt_join_frames},
                            'choice': {'beam_size': 1,
                                       'class': 'choice',
                                       'input_type': 'regression',
                                       'from': ['output'],
                                       'target': 'windowed_data_target',
                                       # 'scheduled_sampling': {'gold_mixin_prob':0.5}
                                       },
                            'pre_slice': {'class': 'slice', 'axis': 'F',
                                          'slice_start': self.audio_opt_feature_size * (self.audio_opt_join_frames - 1),
                                          'from': ['prev:choice']},
                            'pre_net_layer_1': {'activation': 'relu', 'class': 'linear', 'n_out': self.pre_layer1_dim,
                                                'from': ['pre_slice'], 'L2': self.L2_Norm},
                            'pre_net_layer_2': {'activation': 'relu', 'class': 'linear', 'n_out': self.pre_layer2_dim,
                                                'from': ['pre_net_layer_1'], 'dropout': self.pre_layer1_dropout,
                                                'dropout_noise_shape': {'*': None}, 'dropout_on_forward': True,
                                                'L2': self.L2_Norm},
                            'pre_net_layer_2_out': {'class': 'dropout', 'dropout': self.pre_layer2_dropout,
                                                    'from': ['pre_net_layer_2'],
                                                    'dropout_noise_shape': {'*': None}, 'dropout_on_forward': True},
                        }
                        },

            'post_conv_tf': {'class': 'conv', 'activation': None,
                             'from': ['post_conv%i_out' % (len(self.post_conv_dims) - 1)],
                             'filter_size': (5,), 'padding': 'same',
                             'n_out': self.audio_opt_feature_size},
            'dec_output_split': {'class': 'split_dims',
                                 'from': ['decoder'],
                                 'axis': 'F',
                                 'dims': (self.audio_opt_join_frames, -1)},
            'dec_output': {'class': 'merge_dims',
                           'from': ['dec_output_split'],
                           'axes': ['T', 'static:0'],
                           'n_out': self.audio_opt_feature_size, },
            'output': {'class': 'combine', 'kind': 'add', 'loss': 'mean_l1', 'loss_scale': self.post_net_loss_scale,
                       'target': 'padded_data_target',
                       'from': ['dec_output', 'post_conv_tf'],
                       'n_out': self.audio_opt_feature_size},
            'mse_output': {'class': 'copy', 'loss': 'mse', 'loss_scale': 0.0, 'target': 'padded_data_target',
                           'from': ['output'],
                           'n_out': self.audio_opt_feature_size},
            #'stop_token_target': {'class': 'eval',
            #                      'eval': "self.network.get_config().typed_value('_stop_token_target')(source(0, as_data=True))",
            #                      'from': ['windowed_data_target'],
            #                      'register_as_extern_data': 'stop_token_target',
            #                      'out_type': {'shape': (None, 1), 'dim': 1}},
            'stop_token_subnet': stop_token_target_subnet(ramp_length=5),
            'stop_token_target': {'class': 'copy', 'from': 'stop_token_subnet', 'register_as_extern_data': 'stop_token_target'}

        }
        # the bi-directional LSTM part of the encoder. The input is the last convolutional layer, which is added later
        # as convolutional block
        # if the zoneoutlstm type is used, add additional units opts for this
        if self.encoder_lstm_type == "zoneoutlstm":
            encoder_lstm = {
                'lstm0_fw': {'class': 'rec', 'direction': 1,
                             'from': ['embed_conv%i_out' % (len(self.encoder_conv_dims) - 1)],
                             'n_out': self.encoder_lstm_dim, 'unit': 'zoneoutlstm',
                             'unit_opts': {'zoneout_factor_cell': self.zoneout, 'zoneout_factor_output': self.zoneout}},
                'lstm0_bw': {'class': 'rec', 'direction': -1,
                             'from': ['embed_conv%i_out' % (len(self.encoder_conv_dims) - 1)],
                             'n_out': self.encoder_lstm_dim, 'unit': 'zoneoutlstm',
                             'unit_opts': {'zoneout_factor_cell': self.zoneout, 'zoneout_factor_output': self.zoneout}},
            }
        else:
            encoder_lstm = {
                'lstm0_fw': {'class': 'rec', 'direction': 1,
                             'from': ['embed_conv%i_out' % (len(self.encoder_conv_dims) - 1)],
                             'n_out': self.encoder_lstm_dim, 'unit': self.encoder_lstm_type},
                'lstm0_bw': {'class': 'rec', 'direction': -1,
                             'from': ['embed_conv%i_out' % (len(self.encoder_conv_dims) - 1)],
                             'n_out': self.encoder_lstm_dim, 'unit': self.encoder_lstm_type}
            }
        network.update(encoder_lstm)

        # if encoder lstm dropout is used, make sure to add the correct dropout shape
        if self.encoder_lstm_dropout > 0.0 and not self.encoder_lstm_dropout_broadcasting:
            network['encoder']['dropout_noise_shape'] = {'*': None}

        # windowing is only necessary if frames are joined
        #
        if self.audio_opt_join_frames:
            network['windowed_data'] = {
                'class': 'window',
                'from': ['data:audio_features'],
                'window_size': self.audio_opt_join_frames,
                'window_right': self.audio_opt_join_frames - 1,
                'stride': self.audio_opt_join_frames
            }
            network['padded_data_target'] = {
                'class': 'merge_dims',
                'from': ['windowed_data'],
                'axes': ['T', 'static:0'],
                'n_out': self.audio_opt_feature_size,
                'register_as_extern_data': 'padded_data_target',
            }
            network['windowed_data_target'] = {
                'class': 'merge_dims',
                'from': ['windowed_data'],
                'axes': 'static',
                'n_out': self.audio_opt_feature_size * self.audio_opt_join_frames,
                'register_as_extern_data': 'windowed_data_target'
            }
        else:
            network['windowed_data'] = {"class": "copy", "from": ['data:audio_features']}
            network['padded_data_target'] = {"class": "copy", "from": ['data:audio_features'],
                                             'register_as_extern_data': 'padded_data_target'}
            network['windowed_data_target'] = {"class": "copy", "from": ['data:audio_features'],
                                               'register_as_extern_data': 'windowed_data_target'}

        if self.encoder_position_dim:
            # the function creating a positional encoder embedding. to infer the correct shape, an input of the target sequence
            # is needed (lstm0_fw in this case
            network['encoder_position'] = {
                'class': 'positional_encoding', 'from': ['lstm0_fw'], 'n_out': self.encoder_position_dim,
                'out_type': {'dim': self.encoder_position_dim,
                             'shape': (None, self.encoder_position_dim)},
            }
            network['enc_ctx']['from'].append("encoder_position")

        # create the convolutional stacks for the encoder, the first layer uses 'embedding' as input
        assert len(self.encoder_conv_dims) == len(self.encoder_conv_filter_sizes)
        for i, (dim, filter_size) in enumerate(zip(self.encoder_conv_dims, self.encoder_conv_filter_sizes)):
            if i == 0:
                net = conv_block('embed', 0, 'embedding', dim, filter_size, self.L2_Norm, self.encoder_dropout,
                                 regularization=self.encoder_regularization)
            else:
                net = conv_block('embed', i, 'embed_conv%i_out' % (i - 1), dim, filter_size, self.L2_Norm,
                                 self.encoder_dropout,
                                 regularization=self.encoder_regularization)
            network.update(net)

        # create the convolutional stacks for the decoder
        assert len(self.post_conv_dims) == len(self.post_conv_filter_sizes)
        for i, (dim, filter_size) in enumerate(zip(self.post_conv_dims, self.post_conv_filter_sizes)):
            if i == 0:
                net = conv_block('post', 0, 'dec_output', dim, filter_size, self.L2_Norm, self.post_dropout)
            else:
                net = conv_block('post', i, 'post_conv%i_out' % (i - 1), dim, filter_size, self.L2_Norm,
                                 self.post_dropout)

            network.update(net)

        # build the decoder recurrent units and add it to the network
        decoder_cfg = self._decoder()
        network['decoder']['unit'].update(decoder_cfg)

        import copy
        network = copy.deepcopy(network)
        self._lock = True

        return network

    def add_decoding(self, config, dump_input=False, dump_attention=False):
        """
        add the additional layers that are needed for decoding. This inlucdes all HDFDump layers and the additional
        "variable" stop token functions.

        :param dict config: 'network' entry of a config dict
        :return:
        """
        config = copy.deepcopy(config)

        # stop token target is not needed
        dump_input_layer = {'dump_input': {'class': 'hdf_dump', 'filename': 'input.hdf', 'from': ['data:phon_labels'],
                                           'is_output_layer': True}}

        dump_attention_layer = {
            'att_feature_last': {'class': 'swap_axes', 'from': ['att_weights'], 'axis1': -1, 'axis2': -2},
            'dump_att': {'class': 'hdf_dump', 'filename': 'attention.hdf', 'from': ['att_feature_last'],
                         'is_output_layer': True}}

        # we add a variable stop token, that runs a fixed, but definable number of steps after the threshold is exeeded once
        '''
        variable_end_token = {
            'pre_end': {'class': 'compare', 'kind': 'greater', 'from': ['stop_token_sigmoid'], 'value':
                self.decoding_stop_threshold},
            'pre_end_float': {'class': 'cast', 'dtype': 'float32', 'from': ['pre_end']},
            'accum_end': {'class': 'eval',
                          'eval': 'source(0) + tf.minimum(tf.maximum(source(1), source(0)), 1)',
                          'from': ['prev:accum_end', 'pre_end_float'],
                          'out_type': {'dim': 1, 'shape': (1,)}},

            'end_compare': {'class': 'compare', 'kind': 'greater', 'from': ['accum_end'],
                            'value': self.decoding_additional_steps},
            'end': {'class': 'squeeze', 'from': ['end_compare'], 'axis': 'F'}}
        config['network']['decoder']['unit'].update(**variable_end_token)
        '''
        ##config['network'].pop("stop_token_target")
        #config['network']['decoder']['unit']['pre_slice']['from'] = ['prev:output']
        if dump_input:
            config['network'].update(**dump_input_layer)
        if dump_attention:
            config['network']['decoder']['unit'].update(**dump_attention_layer)
        return config

    def add_decoding_forward(self,config,dump_input=False,dump_attention=False, dump_output_layer=False):
        """
        add the additional layers that are needed for decoding. This inlucdes all HDFDump layers and the additional
        "variable" stop token functions
       :param dict config: 'network' entry of a config dict
        :return:
        """
        config = copy.deepcopy(config)

        # stop token target is not needed
        dump_input_layer = {'dump_input': {'class': 'hdf_dump', 'filename': 'input.hdf', 'from': ['data:phon_labels'],
                                           'is_output_layer': True}}
        dump_attention_layer = {
            'att_feature_last': {'class': 'swap_axes', 'from': ['att_weights'], 'axis1': -1, 'axis2': -2},
            'dump_att': {'class': 'hdf_dump', 'filename': 'attention.hdf', 'from': ['att_feature_last'],
                         'is_output_layer': True}}

        # we add a variable stop token, that runs a fixed, but definable number of steps after the threshold is exeeded once

        variable_end_token = {
            'pre_end': {'class': 'compare', 'kind': 'greater', 'from': ['stop_token_sigmoid'], 'value':
                self.decoding_stop_threshold},
            'pre_end_float': {'class': 'cast', 'dtype': 'float32', 'from': ['pre_end']},
            'accum_end': {'class': 'eval',
                          'eval': 'source(0) + tf.minimum(tf.maximum(source(1), source(0)), 1)',
                          'from': ['prev:accum_end', 'pre_end_float'],
                          'out_type': {'dim': 1, 'shape': (1,)}},

            'end_compare': {'class': 'compare', 'kind': 'greater', 'from': ['accum_end'],
                            'value': self.decoding_additional_steps},
            'end': {'class': 'squeeze', 'from': ['end_compare'], 'axis': 'F'}}
        config['network']['decoder']['unit'].update(**variable_end_token)

        config['network'].pop("stop_token_target")
        config['network']['decoder']['unit']['pre_slice']['from'] = ['prev:output']
        if dump_input:
            config['network'].update(**dump_input_layer)
        if dump_attention:
            config['network']['decoder']['unit'].update(**dump_attention_layer)
        return config
