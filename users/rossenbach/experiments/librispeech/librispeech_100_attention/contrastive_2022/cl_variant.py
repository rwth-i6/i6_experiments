import copy

from i6_core.returnn.config import CodeWrapper



def get_contrastive_loss_net(
        loss_scale, softmax_temp, encoder_name, input_mask_name='input_mask', input_name='input', l2=0.0):
    return {
        "class": "subnetwork", "from": [],
        "subnetwork": {
            "enc_masked_frames_": {"class": "masked_computation", "mask": "base:%s" % input_mask_name, "from": "base:%s" % encoder_name,
                                   "unit": {"class": "copy", "from": "data"}},  # [B, T_M, F]
            "enc_masked_frames": {
                "class": "reinterpret_data", "from": "enc_masked_frames_", "set_dim_tags": {"T": CodeWrapper("masked_time_dim")}},

            "c": {"class": "linear", "from": "enc_masked_frames", "out_dim": CodeWrapper("dim_project"), 'L2': l2},

            # We take the non-masked input of the masked frames -> q_t in the paper.
            "input_masked_frames": {"class": "masked_computation", "mask": "base:%s" % input_mask_name, "from": "base:%s" % input_name,
                                    "out_spatial_dim": CodeWrapper("masked_time_dim"),
                                    "unit": {"class": "copy", "from": "data"}},  # [B, T_M, F]
            "q": {"class": "linear", "from": "input_masked_frames", "out_dim": CodeWrapper("dim_project"), 'L2': l2},
            "q_len": {"class": "length", "from": "input_masked_frames", "axis": "T"},  # [B]

            # Candidate samples
            "q_samples_rand_indices__": {
                "class": "rand_int", "maxval": 2 ** 30,
                "from": "input_masked_frames",  # only for masked_time_dim
                "shape": [CodeWrapper("batch_dim"), CodeWrapper("masked_time_dim"), CodeWrapper("dim_neg_samples")]},  # [B, T_M, K] -> 0..BIG
            "q_samples_rand_indices_": {
                "class": "eval", "from": ["q_samples_rand_indices__", "q_len"],
                "eval": "source(0) % tf.maximum(source(1) - 1, 1)"},  # [B, T_M, K] -> 0..T_M-1
            "_range": {"class": "range_in_axis", "from": "input_masked_frames", "axis": CodeWrapper("masked_time_dim")},  # [T_M]
            "_range_ge_indices": {
                "class": "compare", "kind": "greater_equal", "from": ["q_samples_rand_indices_", "_range"]},  # [B, T_M, K]
            "_indices_offsets": {"class": "switch", "condition": "_range_ge_indices", "true_from": 1, "false_from": 0},
            "q_samples_rand_indices": {
                "class": "combine", "kind": "add", "from": ["q_samples_rand_indices_", "_indices_offsets"]},  # [B, T_M, K]
            "q_sampled_frames": {
                "class": "gather", "from": "q",
                "position": "q_samples_rand_indices", "axis": CodeWrapper("masked_time_dim")},  # [B, T_M, K, F]
            "q_expand": {"class": "expand_dims", "axis": "spatial", "dim": CodeWrapper("dim_expand"), "from": "q"},  # [B, T_M ,1, F]
            "Q": {"class": "concat", "from": [
                ("q_expand", CodeWrapper("dim_expand")), ("q_sampled_frames", CodeWrapper("dim_neg_samples"))]},  # [B, T_M, K+1, F]

            # Cosine similarity between sampled frames and masked encoder frames
            "cos_similarity": {
                "class": "subnetwork", "from": ["Q", "c"], "concat_sources": False,
                "subnetwork": {
                    # [B_M, K+1, F] * [B_M, F] -> [B_M, K+1]
                    "dot": {"class": "dot", "from": ["data:0", "data:1"], "reduce": CodeWrapper("dim_project")},
                    "norm_a_sq_": {"class": "eval", "from": "data:0", "eval": "source(0) ** 2"},
                    "norm_a_sq": {"class": "reduce", "mode": "sum", "from": "norm_a_sq_", "axes": CodeWrapper("dim_project")},  # [B, T_M, K+1]
                    "norm_b_sq_": {"class": "eval", "from": "data:1", "eval": "source(0) ** 2"},
                    "norm_b_sq": {"class": "reduce", "mode": "sum", "from": "norm_b_sq_", "axes": CodeWrapper("dim_project")},  # [B, T_M]
                    "output": {
                        "class": "eval", "from": ["dot", "norm_a_sq", "norm_b_sq"],
                        "eval": "source(0) * tf.minimum(tf.math.rsqrt(source(1) * source(2)), 1./1e-8)"},  # [B, T_M, K+1]
                },
            },

            # The contrastive loss is the negative log-likelihood of the softmax of the cosine similarity
            "log_sm_cos_sim": {
                "class": "softmax_over_spatial", "from": "cos_similarity", "axis": CodeWrapper("dim_expand + dim_neg_samples"),
                "log_space": True, "energy_factor": softmax_temp},  # [B, T_M, K+1]
            "log_likelihood": {
                "class": "gather", "from": "log_sm_cos_sim", "axis": CodeWrapper("dim_expand + dim_neg_samples"), "position": 0},  # [B, T_M]
            "neg_los_likelihood": {"class": "eval", "from": "log_likelihood", "eval": "-source(0)"},  # [B, T_M]
            "output": {"class": "copy", "from": "neg_los_likelihood"},
        },
        "loss": "as_is", "loss_scale": loss_scale
    }


def create_cl_variant(contrastive_loss_opts, exp_config):

    # TODO: variant 1 [Default] (covered by upsample option)
    # - upsample encoder (or "c") using transposed conv

    # TODO: Variant 2
    # - create downsampled mask to give to constrastive loss
    # - do average pooling for q to get (q / factor) and use this as "q"
    # - upsample mask for input mask for the ASR network -> resample layer nearest

    # TODO variant 3
    # - create mask and then apply max pool to downsample
    # - do average pooling for q to get (q / factor) and use this as "q"

    # TODO variant 4
    # - create a downsampled mask and apply it only after the downsampling layers (not direcly on input -> no upsample)
    # - do average pooling for q to get q / 6 and use this for CL
    # - use the downsampled mask for CL

    # variant 5
    # - mask on real samples
    # C from first encoder layer before pooling

    # TODO: ideas
    # - use strided conv instead of avg pooling for mapping q -> q/6
    # - create two encoder networks with shared paramters for each ASR loss and CL loss

    if contrastive_loss_opts is None:
        return

    # CL configuration
    variant = contrastive_loss_opts['variant']
    num_neg_samples = contrastive_loss_opts['num_neg_samples']
    softmax_temp = contrastive_loss_opts['softmax_temp']
    contrastive_loss_scale = contrastive_loss_opts['loss_scale']
    masked_input_layer_name = contrastive_loss_opts['masked_input_layer_name']
    next_layer_names = contrastive_loss_opts['next_layer_names']
    masked_input_dim = contrastive_loss_opts['masked_input_dim']
    project_dim = contrastive_loss_opts['project_dim']
    proj_l2 = contrastive_loss_opts.get('l2', 0.0)


    # add dimension tags
    exp_config['masked_time_dim'] = CodeWrapper("SpatialDim(\"masked_time\")")
    exp_config['enc_feat_dim'] = CodeWrapper("FeatureDim(\"encoder_dim\", 2048)")
    exp_config['input_dim'] = CodeWrapper(f"FeatureDim(\"input\", {masked_input_dim})")
    exp_config['dim_project'] = CodeWrapper(f"FeatureDim(\"project_dim\", {project_dim})")
    exp_config['dim_neg_samples'] = CodeWrapper(
        f"SpatialDim(\"neg_samples\", {num_neg_samples})")
    exp_config['dim_expand'] = CodeWrapper("SpatialDim(\"expand_dim\", 1)")

    input_name = 'input'
    input_mask_name = 'input_mask'

    exp_config['network']['input'] = {'class': 'copy', 'from': masked_input_layer_name}

    # True -> should be masked out by mask embed vector., False -> keep
    exp_config['network']['input_mask'] = {
        "class": "eval", "from": masked_input_layer_name,
        "eval": CodeWrapper("get_contrastive_loss_mask"),
        "out_type": {"dtype": "bool", "shape": (None,)}
    }  # [B,T]

    if variant == 1 or variant == 2 or variant == 5:
        # add data time dim tag to sync with resize
        exp_config['data_time_dim'] = CodeWrapper("SpatialDim(\"data_time_dim\")")

        exp_config['network'][masked_input_layer_name + '_'] = \
            copy.deepcopy(exp_config['network'][masked_input_layer_name])
        exp_config['network'][masked_input_layer_name] = {
            'class': 'reinterpret_data', 'from': 'source_', 'set_dim_tags': {'T': CodeWrapper("data_time_dim")}
        }

    if variant == 2 or variant == 3 or variant == 4:

        # used to sync avg_pool time dim with encoder time dim
        exp_config['avg_pool_time_dim'] = CodeWrapper("SpatialDim(\"avg_pool_time_dim\")")

        # average pool q -> q / 6
        exp_config['network']['avg_pool_input_'] = {
            'class': 'pool', 'mode': 'avg', 'from': 'input', 'pool_size': (6,), 'padding': 'same'
        }  # [B, T/6, F]
        exp_config['network']['avg_pool_input'] = {
            'class': 'reinterpret_data', 'from': 'avg_pool_input_', 'set_dim_tags': {'T': CodeWrapper('avg_pool_time_dim')}
        }
        input_name = 'avg_pool_input'

        if variant == 2 or variant == 4:
            # create the downsampled mask
            exp_config['network']['downsample_input_mask'] = {
                "class": "eval", "from": 'avg_pool_input', "eval": CodeWrapper("get_contrastive_loss_mask"),
                "out_type": {"dtype": "bool", "shape": (None,)}
            }  # [B, T/6]
            input_mask_name = 'downsample_input_mask'

            # for variant 4, no need to upsample the mask since we use it after subsampling
            if variant == 2:
                # upsample mask again
                exp_config['network']['input_mask_to_int'] = {'class': 'cast', 'from': 'downsample_input_mask', 'dtype': 'int32'}
                exp_config['network']['input_mask_upsample'] = {
                    'class': 'resize', 'factor': 6, 'axis': 'T', 'kind': 'nn', 'from': 'input_mask_to_int'
                }

                exp_config['network']['input_mask_'] = {'class': 'cast', 'from': 'input_mask_upsample', 'dtype': 'bool'}
                exp_config['network']['input_mask_reinter'] = {
                    'class': 'reinterpret_data', 'from': 'input_mask_', 'set_dim_tags': {'T': CodeWrapper("data_time_dim")}
                }

                # gather due to ceiling
                exp_config['network']['pos'] = {"class": "range_in_axis", "axis": "T", "from": masked_input_layer_name}
                # overwrite here
                exp_config['network']['input_mask'] = {
                    "class": "gather", "position": "pos", "axis": "T", "from": "input_mask_reinter"
                }  # [B,T]

            if variant == 4:
                # overwrite
                exp_config['network']['input_mask'] = copy.deepcopy(exp_config['network']['downsample_input_mask'])

        elif variant == 3:
            # max pool mask
            #
            # 1. we need to cast to float since pooling does not work with bool dtype
            # 2. we need to sync time dim with avg pool q

            exp_config['network']['input_mask_as_float'] = {'class': 'cast', 'from': 'input_mask', 'dtype': 'float32'}
            exp_config['network']['max_pool_input_mask0'] = {"class": "expand_dims", "from": "input_mask_as_float", "axis": "F"}  # [B,T,1]
            exp_config['network']['max_pool_input_mask1'] = {
                'class': 'pool', 'mode': 'max', 'from': 'max_pool_input_mask0', 'pool_size': (6,), 'padding': 'same'
            }
            exp_config['network']['max_pool_input_mask2'] = {"class": "squeeze", "from": "max_pool_input_mask1", "axis": "F"}  # [B,T]
            exp_config['network']['max_pool_input_mask3'] = {"class": "cast", "dtype": "bool", "from": "max_pool_input_mask2"}
            exp_config['network']['max_pool_input_mask'] = {
                "class": "reinterpret_data", "from": "max_pool_input_mask3",
                "set_dim_tags": {"T": CodeWrapper("avg_pool_time_dim")}
            }  # [B,T]
            input_mask_name = 'max_pool_input_mask'

    exp_config['network']['mask_emb'] = {"class": "variable", "shape": [CodeWrapper("input_dim")],
                                         "init": "RandomUniform(0., 1.)"}

    if variant == 4:

        layer_after_downsample = contrastive_loss_opts['layer_after_downsample']

        # reinterpret "layer_after_downsample" to match avg pool time dim
        exp_config['network'][layer_after_downsample + '_'] = copy.deepcopy(exp_config['network'][layer_after_downsample])
        exp_config['network'][layer_after_downsample] = {
            'class': 'reinterpret_data', 'from': layer_after_downsample + '_',
            'set_dim_tags': {'T': CodeWrapper('avg_pool_time_dim')}
        }

        exp_config['network']['input_masked'] = {
            "class": "switch", "condition": "input_mask", "true_from": 'mask_emb',
            "false_from": layer_after_downsample
        }
        next_layer_names = contrastive_loss_opts['next_layers_after_downsample']
    else:
        exp_config['network']['input_masked'] = {
            "class": "switch", "condition": "input_mask", "true_from": 'mask_emb', "false_from": masked_input_layer_name
        }  # [B,T,F]

    # next layers should use the masked input now
    if isinstance(next_layer_names, str):
        next_layer_names = [next_layer_names]
    for l in next_layer_names:
        exp_config['network'][l]['from'] = 'input_masked'

    # reinterpret encoder
    encoder_dim_tags = {"F": CodeWrapper("enc_feat_dim")}
    if variant == 2 or variant == 3:
        encoder_dim_tags['T'] = CodeWrapper('avg_pool_time_dim')

    exp_config['network']['encoder0'] = copy.deepcopy(exp_config['network']['encoder'])
    exp_config['network']['encoder'] = {'class': 'reinterpret_data', 'from': 'encoder0', "set_dim_tags": encoder_dim_tags}

    # upsample encoder in case needed to compute contrastive loss
    if variant == 1:
        exp_config['network']['encoder_upsample_long'] = {
            "class": "transposed_conv", "activation": None, "filter_size": [6], "from": "encoder", "n_out": 2048,
            "strides": [6],
        }
        exp_config['network']['pos'] = {"class": "range_in_axis", "axis": "T", "from": "input_mask"}
        exp_config['network']['encoder_upsample_'] = {
            "class": "gather", "position": "pos", "axis": "T", "from": "encoder_upsample_long"
        }
        exp_config['network']['encoder_upsample'] = {
            'class': 'reinterpret_data', 'from': 'encoder_upsample_', 'set_dim_tags': {"T": CodeWrapper("data_time_dim")}}

    # add contrastive loss subnetwork
    if variant == 1:
        encoder_name = 'encoder_upsample'
    elif variant == 5:
        exp_config['network']['encoder_merge_layer'] = {'class': 'copy', 'from': contrastive_loss_opts["encoder_in_layers"]}
        encoder_name = "encoder_merge_layer"
    else:
        encoder_name = 'encoder'

    exp_config['network']['contrastive_loss'] = get_contrastive_loss_net(
        loss_scale=contrastive_loss_scale, softmax_temp=softmax_temp, encoder_name=encoder_name,
        input_mask_name=input_mask_name, input_name=input_name, l2=proj_l2)